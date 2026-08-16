from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import uuid4

from fakturama_automation.domain.errors import (
    ManualReviewRequired,
    TransientUiError,
    UnsupportedAutomation,
    VerificationError,
)
from fakturama_automation.domain.matching import (
    addresses_match,
    has_distinct_delivery_address,
    main_address_only,
    normalize_text,
)
from fakturama_automation.domain.models import (
    Address,
    Debtor,
    DebtorCandidate,
    DocumentRecord,
    InvoiceSnapshot,
    OrderInput,
    OrderItem,
    OrderSnapshot,
    OrderTotals,
    Payment,
    PaymentMethodCandidate,
    PaymentStatus,
    ProductCandidate,
    VatCandidate,
)
from fakturama_automation.domain.numbers import parse_decimal_text


def _load_pywinauto():
    try:
        from pywinauto import Application, Desktop, mouse
    except ImportError as exc:  # pragma: no cover - optional Windows package
        raise RuntimeError(
            "pywinauto is required for the UIA gateway. Install the 'uia' project extra."
        ) from exc
    return Application, Desktop, mouse


@dataclass(frozen=True)
class UiaProfile:
    values: dict[str, Any]

    @classmethod
    def load(cls, path: Path) -> "UiaProfile":
        return cls(json.loads(path.read_text(encoding="utf-8")))

    @property
    def window_title_re(self) -> str:
        return self.values["window_title_re"]

    @property
    def date_format(self) -> str:
        return self.values.get("date_format", "%Y-%m-%d")

    @property
    def date_segment_order(self) -> tuple[str, str, str]:
        order = tuple(
            self.values.get("date_segment_order", ["month", "day", "year"])
        )
        if len(order) != 3 or set(order) != {"month", "day", "year"}:
            raise UnsupportedAutomation(
                "date_segment_order must contain month, day, and year exactly once"
            )
        return order

    @property
    def startup_timeout_seconds(self) -> float:
        return float(self.values.get("startup_timeout_seconds", 300))

    @property
    def startup_poll_seconds(self) -> float:
        return float(self.values.get("startup_poll_seconds", 0.25))

    @property
    def editor_timeout_seconds(self) -> float:
        """Maximum wait after an editor-opening action has been invoked."""
        return float(self.values.get("editor_timeout_seconds", 20))

    @property
    def startup_ready_names(self) -> list[str]:
        value = self.values.get("startup_ready_names", [])
        return value if isinstance(value, list) else [str(value)]

    def aliases(self, group: str, key: str) -> list[str]:
        try:
            value = self.values[group][key]
        except KeyError as exc:
            raise UnsupportedAutomation(f"profile has no {group}.{key} locator") from exc
        return value if isinstance(value, list) else [value]


class SemanticUiaSession:
    """Small semantic UIA facade with no stored absolute coordinates."""

    CONTROL_INPUTS = ("Edit", "ComboBox", "Spinner", "Document")

    def __init__(
        self,
        executable: Path,
        profile: UiaProfile,
        timeout: float | None = None,
    ) -> None:
        self.executable = executable
        self.profile = profile
        self.timeout = timeout if timeout is not None else profile.startup_timeout_seconds
        self.app = None
        self.main = None

    def attach(self) -> None:
        Application, Desktop, _ = _load_pywinauto()
        # SWT top-level windows can be absent from UIA desktop enumeration even
        # though their descendants are accessible through UIA by native handle.
        # Discover the handle with Win32, then reconnect that handle with UIA.
        desktop = Desktop(backend="win32")
        windows = self._matching_windows(desktop)
        ready = self._ready_windows(Application, windows)
        if len(ready) == 1:
            self.app, self.main, _ = ready[0]
            return
        if len(ready) > 1:
            raise ManualReviewRequired(
                "multiple ready Fakturama windows match "
                f"{self.profile.window_title_re!r}: {self._window_details(ready)}"
            )

        launched_app = None
        if not windows:
            # Eclipse/SWT applications may never report an idle message queue and
            # may create their real window in a Java child process. Start without
            # waiting for idle, then discover the ready window at desktop scope.
            launched_app = Application(backend="uia").start(
                str(self.executable), wait_for_idle=False
            )
            self.app = launched_app

        deadline = time.monotonic() + self.timeout
        last_matches = windows
        while time.monotonic() < deadline:
            last_matches = self._matching_windows(desktop)
            ready = self._ready_windows(Application, last_matches)
            if len(ready) == 1:
                self.app, self.main, _ = ready[0]
                return
            if len(ready) > 1:
                raise ManualReviewRequired(
                    "multiple ready Fakturama windows appeared at startup: "
                    + self._window_details(ready)
                )
            time.sleep(self.profile.startup_poll_seconds)

        launcher_pid = getattr(launched_app, "process", None)
        details = self._window_details(last_matches) if last_matches else "none"
        raise TransientUiError(
            f"Fakturama window matching {self.profile.window_title_re!r} did not "
            f"become ready within {self.timeout:g} seconds; launcher PID="
            f"{launcher_pid or 'unknown'}; matching windows={details}"
        )

    def _matching_windows(self, desktop) -> list:
        try:
            return desktop.windows(
                title_re=self.profile.window_title_re,
                visible_only=True,
                enabled_only=True,
            )
        except Exception:
            return []

    def _ready_windows(self, Application, windows: list) -> list:
        ready = []
        for native_window in windows:
            try:
                app = Application(backend="uia").connect(handle=native_window.handle)
                window = app.window(handle=native_window.handle)
            except Exception:
                continue
            if self._window_is_ready(window):
                ready.append((app, window, native_window))
        return ready

    def _window_is_ready(self, window) -> bool:
        expected = [normalize_text(name) for name in self.profile.startup_ready_names]
        if not expected:
            return True
        try:
            elements = [window, *window.descendants()]
        except Exception:
            return False
        for element in elements:
            try:
                if not element.is_visible() or not element.is_enabled():
                    continue
                actual = normalize_text(element.window_text())
            except Exception:
                continue
            if actual and any(name == actual or name in actual for name in expected):
                return True
        return False

    @staticmethod
    def _window_details(windows: list) -> str:
        details = []
        seen_handles = set()
        for window in windows:
            try:
                handle = window.handle
                if handle in seen_handles:
                    continue
                seen_handles.add(handle)
                details.append(
                    f"{window.window_text()!r} (PID {window.process_id()}, handle {handle})"
                )
            except Exception:
                continue
        return ", ".join(details) or "unavailable"

    def root(self):
        if self.main is None:
            raise RuntimeError("UIA session is not attached")
        return self.main

    def active_window(self):
        _, Desktop, _ = _load_pywinauto()
        windows = Desktop(backend="uia").windows(
            process=self.root().process_id(), visible_only=True, enabled_only=True
        )
        return windows[-1] if windows else self.root()

    def wait_for_named_window(
        self, names: list[str], *, timeout: float | None = None
    ):
        """Return the specifically titled modal instead of an arbitrary process window."""
        deadline = time.monotonic() + (timeout or self.timeout)
        while time.monotonic() < deadline:
            unique = self._visible_named_windows(names)
            if len(unique) == 1:
                return next(iter(unique.values()))
            if len(unique) > 1:
                raise UnsupportedAutomation(
                    f"multiple visible modal windows named {names!r} were found"
                )
            time.sleep(0.2)
        raise TransientUiError(f"timed out waiting for modal window {names!r}")

    def _visible_named_windows(self, names: list[str]) -> dict[str, Any]:
        """Rediscover named windows from the live UI tree, avoiding stale wrappers."""
        _, Desktop, _ = _load_pywinauto()
        expected = {normalize_text(name) for name in names}
        matches = []
        try:
            matches.extend(
                window
                for window in Desktop(backend="uia").windows(
                    process=self.root().process_id(),
                    visible_only=True,
                    enabled_only=True,
                )
                if normalize_text(window.window_text()) in expected
            )
        except Exception:
            pass
        if not matches:
            try:
                matches.extend(
                    element
                    for element in self.root().descendants()
                    if element.element_info.control_type in {"Window", "Pane"}
                    and normalize_text(element.window_text()) in expected
                    and element.is_visible()
                    and element.is_enabled()
                )
            except Exception:
                pass
        return {_runtime_id(match): match for match in matches}

    def find_named(
        self,
        names: list[str],
        *,
        control_types: tuple[str, ...] | None = None,
        scope=None,
        exact: bool = True,
    ):
        scope = scope or self.active_window()
        matches = []
        expected = {normalize_text(name) for name in names}
        for element in scope.descendants():
            try:
                control_type = element.element_info.control_type
                title = element.window_text()
            except Exception:
                continue
            if control_types and control_type not in control_types:
                continue
            normalized = normalize_text(title)
            found = normalized in expected if exact else any(item in normalized for item in expected)
            if found and element.is_visible():
                if control_types == ("Text",) or element.is_enabled():
                    matches.append(element)
        if len(matches) != 1:
            raise UnsupportedAutomation(
                f"expected one accessible control named {names!r}, found {len(matches)}"
            )
        return matches[0]

    def invoke(self, names: list[str], *, scope=None) -> None:
        element = self.find_named(
            names,
            control_types=(
                "Button",
                "SplitButton",
                "Hyperlink",
                "MenuItem",
                "TabItem",
            ),
            scope=scope,
        )
        try:
            element.invoke()
        except Exception:
            element.click_input()

    def invoke_text(self, names: list[str], *, scope=None) -> None:
        element = self.find_named(names, control_types=("Text",), scope=scope)
        try:
            element.invoke()
        except Exception:
            element.click_input()

    def invoke_upper_right_list_plus(self, *, scope=None) -> None:
        """Invoke the documented green ``+`` at the upper-right of a data list.

        Fakturama's SWT lists may expose this icon as a named ``+`` button or as
        an unnamed image.  Discovery is constrained to the upper-right edge of
        the live table; it never falls back to a generic control named ``Add``.
        """
        scope = scope or self.active_window()
        controls = []
        for element in scope.descendants():
            try:
                if element.element_info.control_type not in {
                    "Button",
                    "Image",
                    "SplitButton",
                }:
                    continue
                if not element.is_visible() or not element.is_enabled():
                    continue
                controls.append(element)
            except Exception:
                continue

        named = [
            element
            for element in controls
            if normalize_text(element.window_text()) == "+"
        ]
        if len(named) == 1:
            self._invoke_element(named[0])
            return

        anchors = []
        for element in scope.descendants():
            try:
                if element.element_info.control_type not in {
                    "Header",
                    "HeaderItem",
                    "DataGrid",
                    "Table",
                }:
                    continue
                if element.is_visible():
                    anchors.append(element.rectangle())
            except Exception:
                continue
        if not anchors:
            raise UnsupportedAutomation(
                "could not ground the documented upper-right green '+' to a live data list"
            )

        table_left = min(rect.left for rect in anchors)
        table_right = max(rect.right for rect in anchors)
        table_top = min(rect.top for rect in anchors)
        table_width = max(1, table_right - table_left)
        candidates = []
        for element in named or controls:
            try:
                rect = element.rectangle()
                name = normalize_text(element.window_text())
            except Exception:
                continue
            near_right = rect.left >= table_right - max(80, table_width // 8)
            near_top = table_top - 80 <= rect.top <= table_top + 80
            if near_right and near_top and name in {"", "+"}:
                score = abs(rect.right - table_right) + abs(rect.top - table_top)
                candidates.append((score, element))
        candidates.sort(key=lambda item: item[0])
        if not candidates or (
            len(candidates) > 1 and candidates[0][0] == candidates[1][0]
        ):
            raise UnsupportedAutomation(
                "expected one documented green '+' at the upper-right of the data list"
            )
        self._invoke_element(candidates[0][1])

    @staticmethod
    def _invoke_element(element) -> None:
        try:
            element.invoke()
        except Exception:
            element.click_input()

    def invoke_related_image(
        self, label_names: list[str], *, ordinal: int = 0, scope=None
    ) -> None:
        """Invoke an unnamed SWT image by labeled section and vertical order."""
        scope = scope or self.active_window()
        label = self.find_named(label_names, control_types=("Text",), scope=scope)
        label_rect = label.rectangle()
        label_height = max(1, label_rect.bottom - label_rect.top)
        candidates = []
        for element in scope.descendants():
            try:
                if element.element_info.control_type != "Image" or not element.is_visible():
                    continue
                rect = element.rectangle()
            except Exception:
                continue
            horizontal_overlap = min(label_rect.right, rect.right) - max(
                label_rect.left, rect.left
            )
            below_gap = rect.top - label_rect.bottom
            if horizontal_overlap > 0 and 0 <= below_gap <= 6 * label_height:
                candidates.append((rect.top, abs(rect.left - label_rect.left), element))
        candidates.sort(key=lambda item: item[:2])
        if len(candidates) <= ordinal:
            raise UnsupportedAutomation(
                f"section {label_names!r} exposes no image at ordinal {ordinal}"
            )
        candidates[ordinal][2].click_input()

    def expand_labeled_control(self, label_names: list[str], *, scope=None) -> None:
        """Click the unlabeled expander immediately to the right of a field."""
        scope = scope or self.active_window()
        field = self.input_for_label(label_names, scope=scope)
        field_rect = field.rectangle()
        candidates = []
        for element in scope.descendants():
            try:
                if element.element_info.control_type not in {"Button", "Image"}:
                    continue
                if not element.is_visible() or not element.is_enabled():
                    continue
                rect = element.rectangle()
            except Exception:
                continue
            vertical_overlap = min(field_rect.bottom, rect.bottom) - max(
                field_rect.top, rect.top
            )
            if rect.left >= field_rect.right - 4 and vertical_overlap > 0:
                candidates.append((rect.left - field_rect.right, element))
        candidates.sort(key=lambda pair: pair[0])
        if not candidates or (
            len(candidates) > 1 and candidates[0][0] == candidates[1][0]
        ):
            raise UnsupportedAutomation(
                f"no unique expander is associated with {label_names!r}"
            )
        try:
            candidates[0][1].invoke()
        except Exception:
            candidates[0][1].click_input()

    def read_tab_value(self, tab_names: list[str], *, scope=None) -> str:
        """Select a named address tab and read its associated display field."""
        scope = scope or self.active_window()
        tab = self.find_named(tab_names, control_types=("TabItem",), scope=scope)
        try:
            tab.select()
        except Exception:
            tab.click_input()
        time.sleep(0.1)
        tab_rect = tab.rectangle()
        candidates = []
        for element in scope.descendants():
            try:
                if element.element_info.control_type not in {"Edit", "Document"}:
                    continue
                if not element.is_visible():
                    continue
                rect = element.rectangle()
            except Exception:
                continue
            horizontal_overlap = min(tab_rect.right, rect.right) - max(
                tab_rect.left, rect.left
            )
            gap = rect.top - tab_rect.bottom
            if horizontal_overlap > 0 and 0 <= gap < 100:
                candidates.append((gap, abs(rect.left - tab_rect.left), element))
        candidates.sort(key=lambda item: item[:2])
        if not candidates or (
            len(candidates) > 1 and candidates[0][:2] == candidates[1][:2]
        ):
            raise UnsupportedAutomation(
                f"no unique value field is associated with tab {tab_names!r}"
            )
        return self.read_element_value(candidates[0][2])

    def select_tab(self, names: list[str]) -> None:
        element = self.find_named(names, control_types=("TabItem",))
        try:
            element.select()
        except Exception:
            element.click_input()

    def input_for_label(self, label_names: list[str], *, scope=None):
        scope = scope or self.active_window()
        label = self.find_named(label_names, control_types=("Text",), scope=scope)
        label_rect = label.rectangle()
        candidates = []
        for element in scope.descendants():
            try:
                if element.element_info.control_type not in self.CONTROL_INPUTS:
                    continue
                if not element.is_visible() or not element.is_enabled():
                    continue
                rect = element.rectangle()
            except Exception:
                continue
            vertical_overlap = min(label_rect.bottom, rect.bottom) - max(
                label_rect.top, rect.top
            )
            to_right = rect.left >= label_rect.right - 4 and vertical_overlap > 0
            just_below = rect.top >= label_rect.bottom and rect.top - label_rect.bottom < 80
            if to_right or just_below:
                distance = abs(rect.left - label_rect.right) + abs(rect.top - label_rect.top)
                candidates.append((distance, element))
        if not candidates:
            raise UnsupportedAutomation(f"no editable control is associated with {label_names!r}")
        candidates.sort(key=lambda pair: pair[0])
        if len(candidates) > 1 and candidates[0][0] == candidates[1][0]:
            raise UnsupportedAutomation(f"ambiguous editable control for {label_names!r}")
        return candidates[0][1]

    def set_labeled_value(self, label_names: list[str], value: str, *, scope=None) -> None:
        element = self.input_for_label(label_names, scope=scope)
        control_type = element.element_info.control_type
        try:
            if control_type == "ComboBox":
                self._select_combo_option(element, [value])
            else:
                element.set_edit_text(value)
        except Exception as exc:
            raise UnsupportedAutomation(
                f"could not set field {label_names!r} to {value!r}"
            ) from exc
        actual = self.read_element_value(element)
        if normalize_text(actual) != normalize_text(value):
            raise VerificationError(
                f"read-back for {label_names!r} was {actual!r}, expected {value!r}"
            )

    def set_labeled_row_values(
        self,
        label_names: list[str],
        values: list[str],
        *,
        scope=None,
        exact: bool = True,
    ) -> None:
        """Set the nearest left-to-right inputs associated with a combined label."""
        scope = scope or self.active_window()
        label = self.find_named(
            label_names, control_types=("Text",), scope=scope, exact=exact
        )
        label_rect = label.rectangle()
        candidates = []
        for element in scope.descendants():
            try:
                if element.element_info.control_type != "Edit":
                    continue
                if not element.is_visible() or not element.is_enabled():
                    continue
                rect = element.rectangle()
            except Exception:
                continue
            vertical_overlap = min(label_rect.bottom, rect.bottom) - max(
                label_rect.top, rect.top
            )
            if rect.left >= label_rect.right - 4 and vertical_overlap > 0:
                candidates.append((rect.left, element))
        candidates.sort(key=lambda pair: pair[0])
        if len(candidates) < len(values):
            raise UnsupportedAutomation(
                f"combined label {label_names!r} has {len(candidates)} editable "
                f"fields; expected at least {len(values)}"
            )
        associated = candidates[: len(values)]
        for (_, element), value in zip(associated, values, strict=True):
            try:
                element.set_edit_text(value)
            except Exception as exc:
                raise UnsupportedAutomation(
                    f"could not set a field associated with {label_names!r}"
                ) from exc
            actual = self.read_element_value(element)
            if normalize_text(actual) != normalize_text(value):
                raise VerificationError(
                    f"combined-field read-back was {actual!r}, expected {value!r}"
                )

    def select_combo_by_initial(
        self, label_names: list[str], value: str, *, scope=None
    ) -> None:
        """Focus a dropdown by initial, then select and verify the exact value."""
        element = self.input_for_label(label_names, scope=scope)
        if element.element_info.control_type != "ComboBox":
            raise UnsupportedAutomation(
                f"field {label_names!r} is not exposed as a dropdown"
            )
        requested = value.strip()
        if not requested:
            raise ValueError("dropdown value cannot be empty")
        try:
            element.click_input()
        except Exception:
            pass
        try:
            element.type_keys(requested[0])
            exact_options = [
                option
                for option in self._combo_item_texts(element)
                if normalize_text(option) == normalize_text(requested)
            ]
            if len(exact_options) > 1:
                raise UnsupportedAutomation(
                    f"dropdown contains multiple exact options for {requested!r}"
                )
            element.select(exact_options[0] if exact_options else requested)
            element.type_keys("{ENTER}")
        except Exception as exc:
            raise UnsupportedAutomation(
                f"could not select dropdown value {requested!r} in {label_names!r}"
            ) from exc
        actual = self.read_element_value(element)
        if normalize_text(actual) != normalize_text(requested):
            raise VerificationError(
                f"dropdown read-back for {label_names!r} was {actual!r}, "
                f"expected {requested!r}"
            )

    def set_search_query(self, label_names: list[str], value: str, *, scope=None) -> None:
        """Type into a search box; Fakturama search has no submit button."""
        scope = scope or self.active_window()
        try:
            element = self.input_for_label(label_names, scope=scope)
        except UnsupportedAutomation as label_error:
            candidates = []
            for candidate in scope.descendants():
                try:
                    if candidate.element_info.control_type != "Edit":
                        continue
                    if candidate.is_visible() and candidate.is_enabled():
                        candidates.append(candidate)
                except Exception:
                    continue
            if len(candidates) != 1:
                raise UnsupportedAutomation(
                    "search label is not exposed and the active view contains "
                    f"{len(candidates)} editable fields; expected exactly one"
                ) from label_error
            element = candidates[0]

        try:
            element.click_input()
        except Exception:
            pass
        try:
            element.set_edit_text(value)
        except Exception as exc:
            raise UnsupportedAutomation(
                f"could not type search query {value!r}"
            ) from exc
        actual = self.read_element_value(element)
        if normalize_text(actual) != normalize_text(value):
            raise VerificationError(
                f"search read-back was {actual!r}, expected {value!r}"
            )

    def set_labeled_decimal(
        self, label_names: list[str], value: Decimal, *, scope=None
    ) -> None:
        element = self.input_for_label(label_names, scope=scope)
        try:
            element.set_edit_text(format(value, "f"))
        except Exception as exc:
            raise UnsupportedAutomation(
                f"could not set numeric field {label_names!r} to {value!r}"
            ) from exc
        actual = self.read_element_value(element)
        actual_decimal = parse_decimal_text(actual)
        if actual_decimal is None or actual_decimal != value:
            raise VerificationError(
                f"numeric read-back for {label_names!r} was {actual!r}, "
                f"expected {value!r}"
            )

    def set_labeled_date(
        self,
        label_names: list[str],
        value: date,
        *,
        date_format: str,
        scope=None,
    ) -> None:
        element = self.input_for_label(label_names, scope=scope)
        current = self.read_element_value(element).strip()
        accepted_formats = (
            date_format,
            "%d.%m.%Y",
            "%m/%d/%Y",
            "%d/%m/%Y",
            "%b %d, %Y",
        )
        display_format = date_format
        for candidate_format in dict.fromkeys(accepted_formats):
            try:
                datetime.strptime(current, candidate_format)
            except ValueError:
                continue
            display_format = candidate_format
            break
        displayed_value = value.strftime(display_format)

        # Fakturama exposes its SWT DateTime widget as one Edit, but internally
        # it has three independently selected segments. Select the left-most
        # segment from live control geometry, then enter the configured order.
        keyboard_committed = False
        try:
            rectangle = element.rectangle()
            width = max(1, rectangle.right - rectangle.left)
            height = max(1, rectangle.bottom - rectangle.top)
            element.click_input(coords=(max(2, width // 10), height // 2))
            segment_values = {
                "month": str(value.month),
                "day": str(value.day),
                "year": str(value.year),
            }
            for index, segment in enumerate(self.profile.date_segment_order):
                element.type_keys(segment_values[segment], pause=0.05)
                if index < 2:
                    element.type_keys("{RIGHT}")
            element.type_keys("{TAB}")
            keyboard_committed = True
        except Exception:
            keyboard_committed = False

        if not keyboard_committed:
            try:
                element.set_edit_text(displayed_value)
            except Exception as exc:
                raise UnsupportedAutomation(
                    f"could not set date field {label_names!r}"
                ) from exc

        element = self.input_for_label(label_names, scope=scope)
        actual = self.read_element_value(element).strip()
        for candidate_format in dict.fromkeys(accepted_formats):
            try:
                actual_date = datetime.strptime(actual, candidate_format).date()
            except ValueError:
                continue
            if actual_date == value:
                return
            break
        raise VerificationError(
            f"date read-back for {label_names!r} was {actual!r}, "
            f"expected {value.isoformat()!r}"
        )

    def select_semantic_option(
        self,
        label_names: list[str],
        option_names: list[str],
        *,
        distinguishing_options: list[str] | None = None,
        scope=None,
    ) -> None:
        """Select a dropdown option by meaning, with a label-independent fallback.

        SWT sometimes exposes the ComboBox and its values but not the adjacent
        label. The fallback identifies exactly one visible ComboBox containing
        the requested option and the configured sibling options. It uses no
        stored screen coordinates.
        """
        scope = scope or self.active_window()
        try:
            element = self.input_for_label(label_names, scope=scope)
            if element.element_info.control_type != "ComboBox":
                raise UnsupportedAutomation(
                    f"field {label_names!r} is not exposed as a ComboBox"
                )
        except UnsupportedAutomation:
            expected = option_names + list(distinguishing_options or [])
            element = self._unique_combo_with_options(
                option_names, expected, scope=scope
            )
        selected = self._select_combo_option(element, option_names)
        actual = self.read_element_value(element)
        if normalize_text(actual) != normalize_text(selected):
            raise VerificationError(
                f"dropdown read-back was {actual!r}, expected {selected!r}"
            )

    def select_optional_labeled_option(
        self, label_names: list[str], option_names: list[str], *, scope=None
    ) -> bool:
        """Select one exact option, returning false only when that option is absent."""
        element = self.input_for_label(label_names, scope=scope)
        if element.element_info.control_type != "ComboBox":
            raise UnsupportedAutomation(f"field {label_names!r} is not a ComboBox")
        try:
            self._select_combo_option(element, option_names)
            return True
        except UnsupportedAutomation:
            return False

    def select_optional_semantic_option(
        self, label_names: list[str], option_names: list[str], *, scope=None
    ) -> bool:
        """Select an exact option even when SWT omits the adjacent label."""
        scope = scope or self.active_window()
        try:
            return self.select_optional_labeled_option(
                label_names, option_names, scope=scope
            )
        except UnsupportedAutomation as label_error:
            try:
                element = self._unique_combo_with_options(
                    option_names, option_names, scope=scope
                )
            except UnsupportedAutomation as combo_error:
                if "no accessible ComboBox" in str(combo_error):
                    return False
                raise combo_error from label_error
            try:
                self._select_combo_option(element, option_names)
                return True
            except UnsupportedAutomation:
                return False

    def read_semantic_option(
        self,
        label_names: list[str],
        *,
        identifying_options: list[str],
        scope=None,
    ) -> str:
        """Read a ComboBox value using the same semantic discovery rules."""
        scope = scope or self.active_window()
        try:
            element = self.input_for_label(label_names, scope=scope)
            if element.element_info.control_type != "ComboBox":
                raise UnsupportedAutomation(
                    f"field {label_names!r} is not exposed as a ComboBox"
                )
        except UnsupportedAutomation:
            element = self._unique_combo_with_options(
                identifying_options, identifying_options, scope=scope
            )
        return self.read_element_value(element)

    def set_toggle(self, names: list[str], selected: bool, *, scope=None) -> None:
        element = self.find_named(
            names, control_types=("CheckBox", "Button"), scope=scope
        )
        current = self._toggle_state(element)
        if current is None:
            raise UnsupportedAutomation(
                f"control {names!r} does not expose a readable toggle state"
            )
        if current != selected:
            try:
                element.toggle()
            except Exception:
                element.click_input()
        actual = self._toggle_state(element)
        if actual != selected:
            raise VerificationError(
                f"toggle read-back for {names!r} was {actual!r}, expected {selected!r}"
            )

    def invoke_in_group(
        self, group_names: list[str], action_names: list[str], *, scope=None
    ) -> None:
        """Invoke the action closest to a named UI group, avoiding toolbar twins."""
        scope = scope or self.active_window()
        group = self.find_named(
            group_names, control_types=("Text", "Group"), scope=scope, exact=False
        )
        expected = {normalize_text(name) for name in action_names}
        group_rect = group.rectangle()
        candidates = []
        for element in scope.descendants():
            try:
                if element.element_info.control_type not in {
                    "Button",
                    "SplitButton",
                    "Hyperlink",
                    "MenuItem",
                }:
                    continue
                if not element.is_visible() or not element.is_enabled():
                    continue
                if normalize_text(element.window_text()) not in expected:
                    continue
                rect = element.rectangle()
                distance = abs(rect.mid_point().x - group_rect.mid_point().x) + abs(
                    rect.mid_point().y - group_rect.mid_point().y
                )
                candidates.append((distance, element))
            except Exception:
                continue
        candidates.sort(key=lambda item: item[0])
        if not candidates or (
            len(candidates) > 1 and candidates[0][0] == candidates[1][0]
        ):
            raise UnsupportedAutomation(
                f"expected one action {action_names!r} in group {group_names!r}"
            )
        element = candidates[0][1]
        try:
            element.invoke()
        except Exception:
            element.click_input()

    def _unique_combo_with_options(
        self, option_names: list[str], expected_options: list[str], *, scope
    ):
        targets = {normalize_text(value) for value in option_names}
        expected = {normalize_text(value) for value in expected_options}
        candidates = []
        fallback_comboes = []
        for element in scope.descendants():
            try:
                if element.element_info.control_type != "ComboBox":
                    continue
                if not element.is_visible() or not element.is_enabled():
                    continue
                fallback_comboes.append(element)
                items = {normalize_text(item) for item in self._combo_item_texts(element)}
            except Exception:
                continue
            if not items.intersection(expected):
                continue
            score = len(items.intersection(expected))
            candidates.append((score, element))
        if not candidates:
            for element in fallback_comboes:
                try:
                    val = normalize_text(self.read_element_value(element))
                    if val in expected or val in targets:
                        candidates.append((1, element))
                except Exception:
                    continue
        if not candidates:
            if len(fallback_comboes) == 1:
                return fallback_comboes[0]
            raise UnsupportedAutomation(
                f"no accessible ComboBox contains option {option_names!r}"
            )
        best_score = max(score for score, _ in candidates)
        best = [element for score, element in candidates if score == best_score]
        if len(best) != 1:
            raise UnsupportedAutomation(
                f"option {option_names!r} appears in {len(best)} equally matching ComboBoxes"
            )
        return best[0]

    def _select_combo_option(self, element, option_names: list[str]) -> str:
        expected = {normalize_text(value) for value in option_names}
        current_val = self.read_element_value(element)
        if normalize_text(current_val) in expected:
            return current_val

        handle = getattr(element, "handle", None) or getattr(
            getattr(element, "element_info", None), "handle", None
        )
        if handle:
            try:
                import ctypes

                user32 = ctypes.windll.user32
                CB_FINDSTRINGEXACT = 0x0158
                CB_SETCURSEL = 0x014E
                WM_COMMAND = 0x0111
                CBN_SELCHANGE = 1
                for option in option_names:
                    idx = user32.SendMessageW(handle, CB_FINDSTRINGEXACT, -1, option)
                    if idx >= 0:
                        user32.SendMessageW(handle, CB_SETCURSEL, idx, 0)
                        parent_handle = user32.GetParent(handle)
                        ctrl_id = user32.GetDlgCtrlID(handle)
                        wparam = (CBN_SELCHANGE << 16) | (ctrl_id & 0xFFFF)
                        user32.SendMessageW(parent_handle, WM_COMMAND, wparam, handle)
                        time.sleep(0.05)
                        actual = self.read_element_value(element)
                        if normalize_text(actual) == normalize_text(option):
                            return option
            except Exception:
                pass

        try:
            element.click_input()
            time.sleep(0.1)
        except Exception:
            pass

        matches = [
            item
            for item in self._combo_item_texts(element)
            if normalize_text(item) in expected
        ]
        if len(matches) > 1:
            raise UnsupportedAutomation(
                f"expected one dropdown option {option_names!r}, found {len(matches)}"
            )
        if len(matches) == 1:
            try:
                element.select(matches[0])
                actual = self.read_element_value(element)
                if normalize_text(actual) == normalize_text(matches[0]):
                    return matches[0]
            except Exception:
                pass

        for option in option_names:
            try:
                element.select(option)
                actual = self.read_element_value(element)
                if normalize_text(actual) == normalize_text(option):
                    return option
            except Exception:
                pass

        try:
            try:
                element.set_focus()
            except Exception:
                pass
            element.type_keys("{HOME}")
            time.sleep(0.05)
            val = self.read_element_value(element)
            for _ in range(40):
                if normalize_text(val) in expected:
                    return val
                element.type_keys("{DOWN}")
                time.sleep(0.05)
                new_val = self.read_element_value(element)
                if normalize_text(new_val) == normalize_text(val):
                    break
                val = new_val
        except Exception:
            pass

        for option in option_names:
            try:
                element.type_keys(option, with_spaces=True)
                time.sleep(0.05)
                actual = self.read_element_value(element)
                if normalize_text(actual) == normalize_text(option):
                    return option
            except Exception:
                pass

        actual = self.read_element_value(element)
        if normalize_text(actual) in expected:
            return actual

        raise UnsupportedAutomation(
            f"expected one selectable dropdown option {option_names!r}, "
            f"current value is {actual!r}"
        )

    @staticmethod
    def _combo_item_texts(element) -> list[str]:
        values = []
        handle = getattr(element, "handle", None) or getattr(
            getattr(element, "element_info", None), "handle", None
        )
        if handle:
            try:
                import ctypes

                user32 = ctypes.windll.user32
                CB_GETCOUNT = 0x0146
                CB_GETLBTEXT = 0x0148
                CB_GETLBTEXTLEN = 0x0149
                count = user32.SendMessageW(handle, CB_GETCOUNT, 0, 0)
                if 0 < count <= 500:
                    win32_items = []
                    for i in range(count):
                        length = user32.SendMessageW(handle, CB_GETLBTEXTLEN, i, 0)
                        if length >= 0:
                            buf = ctypes.create_unicode_buffer(length + 1)
                            if user32.SendMessageW(handle, CB_GETLBTEXT, i, buf) >= 0:
                                win32_items.append(buf.value)
                    if win32_items:
                        values.extend(win32_items)
            except Exception:
                pass

        try:
            values.extend(str(item) for item in element.item_texts())
        except Exception:
            try:
                values.extend(str(item.window_text()) for item in element.children())
            except Exception:
                pass

        if not values:
            try:
                try:
                    element.expand()
                    time.sleep(0.05)
                except Exception:
                    pass
                try:
                    values.extend(str(item) for item in element.item_texts())
                except Exception:
                    pass
            except Exception:
                pass

        if not values:
            for container in [element, getattr(element, "parent", lambda: None)(), * []]:
                if container is None:
                    continue
                try:
                    descendants = list(container.descendants())
                except Exception:
                    descendants = []
                for candidate in descendants:
                    try:
                        if candidate is element:
                            continue
                        if candidate.element_info.control_type not in {"Text", "ListItem"}:
                            continue
                        if not candidate.is_visible():
                            continue
                        text = str(candidate.window_text()).strip()
                    except Exception:
                        continue
                    if text:
                        values.append(text)
                if values:
                    break

        if not values:
            try:
                _, Desktop, _ = _load_pywinauto()
                desktop = Desktop(backend="uia")
                process_id = getattr(getattr(element, "element_info", None), "process_id", None)
                for win in desktop.windows(process=process_id, visible_only=True):
                    try:
                        for child in win.descendants():
                            if child.element_info.control_type in {"ListItem", "Text"}:
                                txt = str(child.window_text()).strip()
                                if txt:
                                    values.append(txt)
                    except Exception:
                        continue
                    if values:
                        break
            except Exception:
                pass

        try:
            legacy_value = str(element.legacy_properties().get("Value") or "")
            if legacy_value:
                values.append(legacy_value)
        except Exception:
            pass
        return list(dict.fromkeys(values))

    def read_labeled_value(self, label_names: list[str], *, scope=None) -> str:
        return self.read_element_value(self.input_for_label(label_names, scope=scope))

    @staticmethod
    def read_element_value(element) -> str:
        try:
            value = str(element.get_value())
            if value.strip():
                return value
        except Exception:
            pass
        try:
            value = str(element.legacy_properties().get("Value") or "")
            if value.strip():
                return value
        except Exception:
            pass
        try:
            value = str(element.selected_text())
            if value.strip():
                return value
        except Exception:
            pass
        try:
            return str(element.window_text())
        except Exception:
            return ""

    @staticmethod
    def _toggle_state(element) -> bool | None:
        for method_name in ("get_toggle_state", "is_checked"):
            try:
                value = getattr(element, method_name)()
            except Exception:
                continue
            if isinstance(value, bool):
                return value
            if value in (0, 1):
                return bool(value)
        return None

    def wait_for_text(self, names: list[str], timeout: float | None = None) -> None:
        expected = [normalize_text(name) for name in names]
        deadline = time.monotonic() + (timeout or self.timeout)
        while time.monotonic() < deadline:
            for element in self.active_window().descendants():
                try:
                    actual = normalize_text(element.window_text())
                    if actual and any(value in actual for value in expected):
                        return
                except Exception:
                    continue
            time.sleep(0.2)
        raise TransientUiError(f"timed out waiting for text {names!r}")

    def scroll_until_label(
        self,
        label_names: list[str],
        *,
        anchor_names: list[str],
        direction: str,
        scope=None,
        timeout: float = 5,
        exact: bool = True,
    ):
        """Scroll a form at a visible anchor until an off-screen label appears."""
        scope = scope or self.active_window()
        if direction not in {"up", "down"}:
            raise ValueError("scroll direction must be 'up' or 'down'")
        try:
            return self.find_named(
                label_names,
                control_types=("Text",),
                scope=scope,
                exact=exact,
            )
        except UnsupportedAutomation:
            pass

        expected = {normalize_text(name) for name in label_names}
        offscreen = []
        for element in scope.descendants():
            try:
                if element.element_info.control_type != "Text":
                    continue
                actual = normalize_text(element.window_text())
                matches = (
                    actual in expected
                    if exact
                    else any(name in actual for name in expected)
                )
                if matches:
                    offscreen.append(element)
            except Exception:
                continue
        if len(offscreen) == 1:
            try:
                offscreen[0].scroll_into_view()
                time.sleep(0.2)
                return self.find_named(
                    label_names,
                    control_types=("Text",),
                    scope=scope,
                    exact=exact,
                )
            except Exception:
                pass

        anchor = self.find_named(anchor_names, control_types=("Text",), scope=scope)
        point = anchor.rectangle().mid_point()
        _, _, mouse = _load_pywinauto()
        deadline = time.monotonic() + timeout
        wheel_distance = 5 if direction == "up" else -5
        while time.monotonic() < deadline:
            mouse.scroll(coords=(point.x, point.y), wheel_dist=wheel_distance)
            time.sleep(0.2)
            try:
                return self.find_named(
                    label_names,
                    control_types=("Text",),
                    scope=scope,
                    exact=exact,
                )
            except UnsupportedAutomation:
                continue
        raise UnsupportedAutomation(
            f"could not reveal label {label_names!r} by scrolling {direction}"
        )

    def wait_for_control(
        self,
        names: list[str],
        *,
        control_types: tuple[str, ...],
        timeout: float | None = None,
        scope=None,
        exclude_runtime_ids: set[str] | None = None,
    ):
        """Wait for one exact semantic control, not merely matching text."""
        expected = {normalize_text(name) for name in names}
        deadline = time.monotonic() + (timeout or self.timeout)
        while time.monotonic() < deadline:
            current_scope = scope or self.active_window()
            matches = []
            for element in current_scope.descendants():
                try:
                    if element.element_info.control_type not in control_types:
                        continue
                    if normalize_text(element.window_text()) not in expected:
                        continue
                    if exclude_runtime_ids and _runtime_id(element) in exclude_runtime_ids:
                        continue
                    if element.is_visible() and element.is_enabled():
                        matches.append(element)
                except Exception:
                    continue
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                raise UnsupportedAutomation(
                    f"multiple ready controls named {names!r} were found"
                )
            time.sleep(0.2)
        raise TransientUiError(
            f"timed out waiting for {control_types!r} named {names!r}"
        )

    def data_rows(self, *, scope=None) -> list:
        scope = scope or self.active_window()
        return [
            row
            for row in scope.descendants(control_type="DataItem")
            if row.is_visible() and row.is_enabled()
        ]

    def wait_for_stable_data_rows(
        self,
        *,
        scope=None,
        stable_seconds: float = 0.5,
        timeout: float | None = None,
    ) -> list:
        """Wait until the visible list result signature stops changing."""
        scope = scope or self.active_window()
        deadline = time.monotonic() + (timeout or self.timeout)
        previous = None
        stable_since = None
        while time.monotonic() < deadline:
            rows = self.data_rows(scope=scope)
            signature = tuple(
                (_runtime_id(row), tuple(_row_values(row))) for row in rows
            )
            now = time.monotonic()
            if signature == previous:
                if stable_since is None:
                    stable_since = now
                if now - stable_since >= stable_seconds:
                    return rows
            else:
                previous = signature
                stable_since = now
            time.sleep(0.1)
        raise TransientUiError("data-list results did not stabilize before timeout")

    def wait_until_closed(self, element, *, timeout: float | None = None) -> None:
        try:
            names = [element.window_text()]
        except Exception as exc:
            raise UnsupportedAutomation(
                "cannot verify dialog closure without its accessible title"
            ) from exc
        if not normalize_text(names[0]):
            raise UnsupportedAutomation(
                "cannot verify dialog closure because its accessible title is empty"
            )
        deadline = time.monotonic() + (timeout or self.timeout)
        absent_polls = 0
        while time.monotonic() < deadline:
            if not self._visible_named_windows(names):
                absent_polls += 1
                if absent_polls >= 2:
                    return
            else:
                absent_polls = 0
            time.sleep(0.1)
        raise TransientUiError(
            f"dialog {names[0]!r} did not close before timeout"
        )

    def row_value_for_header(
        self,
        row,
        header_names: list[str],
        *,
        scope=None,
        allow_empty: bool = False,
    ) -> str:
        """Read a visible table cell by its live column-header geometry."""
        scope = scope or self.active_window()
        header = self.find_named(
            header_names, control_types=("HeaderItem",), scope=scope
        )
        header_rect = header.rectangle()
        row_rect = row.rectangle()
        candidates = []
        for element in row.descendants():
            try:
                text = element.window_text().strip()
                rect = element.rectangle()
                if (not text and not allow_empty) or not element.is_visible():
                    continue
            except Exception:
                continue
            horizontal_overlap = min(header_rect.right, rect.right) - max(
                header_rect.left, rect.left
            )
            vertical_overlap = min(row_rect.bottom, rect.bottom) - max(
                row_rect.top, rect.top
            )
            if horizontal_overlap <= 0 or vertical_overlap <= 0:
                continue
            area = max(1, (rect.right - rect.left) * (rect.bottom - rect.top))
            candidates.append((-horizontal_overlap, area, text))
        if not candidates:
            raise UnsupportedAutomation(
                f"row exposes no readable cell under header {header_names!r}"
            )
        candidates.sort()
        best = candidates[0]
        tied = {text for overlap, area, text in candidates if (overlap, area) == best[:2]}
        if len(tied) != 1:
            raise UnsupportedAutomation(
                f"row has ambiguous cells under header {header_names!r}"
            )
        return best[2]

    def set_grid_cell(self, row, header_names: list[str], value: str, *, scope=None) -> None:
        """Edit a cell using live header/row bounds, never fixed screen coordinates."""
        _, _, mouse = _load_pywinauto()
        scope = scope or self.active_window()
        header = self.find_named(
            header_names,
            control_types=("HeaderItem", "Text"),
            scope=scope,
        )
        header_rect = header.rectangle()
        row_rect = row.rectangle()
        x = header_rect.mid_point().x
        y = row_rect.mid_point().y
        mouse.double_click(coords=(x, y))
        time.sleep(0.1)
        try:
            focused = self.active_window().get_focus()
            focused.set_edit_text(value)
            focused.type_keys("{ENTER}")
        except Exception as exc:
            raise UnsupportedAutomation(
                f"could not edit grid column {header_names!r} using exposed bounds"
            ) from exc


class PywinautoFakturamaGateway:
    """Fail-closed Windows UIA adapter for Fakturama 2.x.

    The checked-in profile contains label aliases, not coordinates. Run the
    `inspect-uia` CLI command against the target installation before enabling
    writes, then adjust the profile if the installed locale exposes different
    accessible names.
    """

    def __init__(self, executable: Path, profile_path: Path) -> None:
        self.profile = UiaProfile.load(profile_path)
        self.session = SemanticUiaSession(executable, self.profile)
        self.expected_order: OrderInput | None = None
        self.order_number: str | None = None
        self.invoice_number: str | None = None
        self.transaction_id: str | None = None
        self.order_document: DocumentRecord | None = None
        self.invoice_payment: Payment | None = None
        self._records: dict[str, Any] = {}
        self._selected_debtor: Debtor | None = None
        self._order_rows: list[Any] = []
        self._order_row_ids_before_selector: set[str] = set()
        self._product_skus: dict[str, str] = {}
        self._pending_debtor: Debtor | None = None
        self._last_invoice_snapshot: InvoiceSnapshot | None = None
        self._order_editor_tab_id: str | None = None
        self._invoice_editor_tab_id: str | None = None
        self._debtor_editor_tab_id: str | None = None
        self._selector_dialog = None

    def preflight(self) -> None:
        if not self.session.executable.is_file():
            raise FileNotFoundError(self.session.executable)
        self.session.attach()

    def open_new_order(self) -> str:
        existing = self._tab_runtime_ids(self.profile.aliases("titles", "new_order"))
        if existing:
            raise ManualReviewRequired(
                "Fakturama already has an unsaved 'New Order' editor open. "
                "Close that tab (discard it only if it is safe), then rerun; "
                "automation refuses to overwrite or reuse an existing draft."
            )
        self.session.invoke(self.profile.aliases("actions", "new_order"))
        tab = self.session.wait_for_control(
            self.profile.aliases("titles", "new_order"),
            control_types=("TabItem",),
            scope=self.session.root(),
            exclude_runtime_ids=existing,
            timeout=self.profile.editor_timeout_seconds,
        )
        try:
            tab.select()
        except Exception:
            tab.click_input()
        self._order_editor_tab_id = _runtime_id(tab)
        self.order_number = self.session.read_labeled_value(
            self.profile.aliases("labels", "document_number")
        )
        self.transaction_id = uuid4().hex
        return self.order_number

    def set_order_header(self, order: OrderInput) -> None:
        self.expected_order = order
        self.session.set_labeled_date(
            self.profile.aliases("labels", "date"),
            order.order_date,
            date_format=self.profile.date_format,
        )
        self.session.set_labeled_value(
            self.profile.aliases("labels", "external_reference"),
            order.external_reference,
        )
        self.session.select_semantic_option(
            self.profile.aliases("labels", "order_price_mode"),
            self.profile.aliases("choices", "net"),
            distinguishing_options=self.profile.aliases("choices", "gross"),
        )
        self.session.select_semantic_option(
            self.profile.aliases("labels", "order_vat_mode"),
            self.profile.aliases("choices", "with_vat"),
            distinguishing_options=self.profile.aliases("choices", "without_vat"),
        )

    def search_debtors(self, company_or_name: str) -> list[DebtorCandidate]:
        self._activate_order_editor()
        self.session.invoke_related_image(
            self.profile.aliases("labels", "addresses"), ordinal=0
        )
        dialog = self.session.wait_for_named_window(
            self.profile.aliases("titles", "select_address"),
            timeout=self.profile.editor_timeout_seconds,
        )
        self._selector_dialog = dialog
        self.session.set_search_query(
            self.profile.aliases("labels", "search"), company_or_name, scope=dialog
        )
        rows = self.session.wait_for_stable_data_rows(
            scope=dialog,
            stable_seconds=float(
                self.profile.values.get("search_stabilization_seconds", 0.5)
            ),
            timeout=self.profile.editor_timeout_seconds,
        )
        output = []
        for row in rows:
            candidate = self._debtor_candidate_from_row(row, dialog)
            if candidate:
                output.append(candidate)
        return output

    def cancel_active_dialog(self) -> None:
        dialog = (
            self._selector_dialog
            if self._selector_dialog is not None
            else self.session.active_window()
        )
        self._cancel_dialog(dialog)
        self._selector_dialog = None

    def select_debtor(self, record_id: str) -> None:
        row = self._records.get(record_id)
        if row is None:
            raise KeyError(record_id)
        try:
            row.select()
        except Exception:
            row.click_input()
        dialog = (
            self._selector_dialog
            if self._selector_dialog is not None
            else self.session.active_window()
        )
        self.session.invoke(self.profile.aliases("actions", "ok"), scope=dialog)
        self.session.wait_until_closed(
            dialog, timeout=self.profile.editor_timeout_seconds
        )
        self._selector_dialog = None
        expected = self._require_order().debtor
        self._assert_labeled_address(
            "invoice_address", expected.billing_address, expected.company
        )
        if not has_distinct_delivery_address(expected):
            self._assert_labeled_address(
                "delivery_address", expected.effective_delivery_address, expected.company
            )
        self._selected_debtor = main_address_only(expected)

    def read_selected_debtor(self) -> Debtor:
        if self._selected_debtor is None:
            raise VerificationError("no verified Debtor is selected")
        return self._selected_debtor

    def search_payment_methods(self, name: str) -> list[PaymentMethodCandidate]:
        self._open_data_view("payment_methods")
        return self._search_simple_rows(name, PaymentMethodCandidate, "payment")

    def create_payment_method(self, payment: PaymentMethodCandidate) -> str:
        try:
            self.session.invoke(self.profile.aliases("actions", "new_payment_method"))
        except UnsupportedAutomation:
            self.session.invoke_upper_right_list_plus(scope=self.session.active_window())
        self.session.set_labeled_value(self.profile.aliases("labels", "name"), payment.name)
        self.session.set_labeled_value(
            self.profile.aliases("labels", "description"), payment.description
        )
        self.session.select_semantic_option(
            self.profile.aliases("labels", "payment_code"),
            [payment.payment_code],
            distinguishing_options=[
                "Credit transfer",
                "Credit card",
                "SEPA direct debit",
                "Cash",
                "Debit",
            ],
        )
        for key, value in (
            ("cash_discount", "0"),
            ("discount_days", "0"),
            ("net_days", "0"),
        ):
            self.session.set_labeled_decimal(
                self.profile.aliases("labels", key), Decimal(value)
            )
        self._save_once()
        return f"payment-{uuid4().hex}"

    def open_new_debtor(self, debtor: Debtor) -> None:
        existing = self._tab_runtime_ids(self.profile.aliases("titles", "new_debtor"))
        if existing:
            raise ManualReviewRequired(
                "Fakturama already has an unsaved 'New Debtor' editor open. "
                "Close that tab (discard it only if it is safe), then rerun; "
                "automation refuses to overwrite or reuse an existing draft."
            )
        self.session.invoke_text(self.profile.aliases("actions", "new_contact"))
        tab = self.session.wait_for_control(
            self.profile.aliases("titles", "new_debtor"),
            control_types=("TabItem",),
            scope=self.session.root(),
            exclude_runtime_ids=existing,
            timeout=self.profile.editor_timeout_seconds,
        )
        self._debtor_editor_tab_id = _runtime_id(tab)
        # Take-home specification 2.6: leave the proposed Customer ID unchanged.
        if debtor.company:
            self.session.set_labeled_value(
                self.profile.aliases("labels", "company"), debtor.company
            )
        if debtor.first_name or debtor.last_name:
            self.session.set_labeled_row_values(
                self.profile.aliases("labels", "debtor_name_row"),
                [debtor.first_name or "", debtor.last_name or ""],
            )
        if debtor.salutation:
            self.session.select_semantic_option(
                self.profile.aliases("labels", "salutation"), [debtor.salutation]
            )
        else:
            actual_salutation = self.session.read_semantic_option(
                self.profile.aliases("labels", "salutation"),
                identifying_options=["---"],
            )
            if actual_salutation.strip() != "---":
                raise VerificationError(
                    f"Salutation is {actual_salutation!r}, expected the proposed '---'"
                )
        self._fill_address(debtor.billing_address)
        self._expand_address_roles()
        self.session.set_toggle(
            self.profile.aliases("actions", "invoice_address_role"), True
        )
        delivery = debtor.delivery_address
        if delivery is None or addresses_match(debtor.billing_address, delivery):
            self.session.set_toggle(
                self.profile.aliases("actions", "delivery_address_role"), True
            )
        self.session.select_tab(self.profile.aliases("tabs", "miscellaneous"))
        if debtor.alias:
            self.session.set_labeled_value(
                self.profile.aliases("labels", "alias"), debtor.alias
            )
        self.session.set_labeled_decimal(
            self.profile.aliases("labels", "discount"), Decimal("0")
        )
        self.session.select_semantic_option(
            self.profile.aliases("labels", "debtor_price_mode"),
            self.profile.aliases("choices", "net"),
            distinguishing_options=self.profile.aliases("choices", "gross"),
        )
        try:
            self.session.select_tab(self.profile.aliases("tabs", "payment"))
        except UnsupportedAutomation:
            pass
        self._pending_debtor = main_address_only(debtor)

    def _debtor_fallback_names(self) -> list[str]:
        names = list(self.profile.aliases("titles", "new_debtor"))
        if self._pending_debtor:
            company = getattr(self._pending_debtor, "company", None)
            if company:
                names.append(company)
            first = getattr(self._pending_debtor, "first_name", "") or ""
            last = getattr(self._pending_debtor, "last_name", "") or ""
            full_name = f"{first} {last}".strip()
            if full_name:
                names.append(full_name)
            if last:
                names.append(last)
            if first:
                names.append(first)
        return list(dict.fromkeys(names))

    def select_debtor_payment_method(self, payment_method: str) -> bool:
        if self._pending_debtor is None:
            raise RuntimeError("no Debtor editor is open")
        self._debtor_editor_tab_id = self._activate_editor_tab(
            self._debtor_editor_tab_id,
            fallback_names=self._debtor_fallback_names(),
        )
        try:
            self.session.select_tab(self.profile.aliases("tabs", "payment"))
        except UnsupportedAutomation:
            pass
        if self.session.select_optional_semantic_option(
            self.profile.aliases("labels", "payment_method"), [payment_method]
        ):
            return True
        return self.session.select_optional_semantic_option(["Payment"], [payment_method])

    def save_debtor(self) -> str:
        if self._pending_debtor is None:
            raise RuntimeError("no Debtor editor is open")
        self._debtor_editor_tab_id = self._activate_editor_tab(
            self._debtor_editor_tab_id,
            fallback_names=self._debtor_fallback_names(),
        )
        self._save_once()
        self._pending_debtor = None
        return f"debtor-{uuid4().hex}"

    def search_vats(self, name: str) -> list[VatCandidate]:
        self._open_data_view("vats")
        return self._search_simple_rows(name, VatCandidate, "vat")

    def create_vat(self, vat: VatCandidate) -> str:
        self.session.invoke_upper_right_list_plus(scope=self.session.active_window())
        self.session.set_labeled_value(self.profile.aliases("labels", "name"), vat.name)
        self.session.set_labeled_value(
            self.profile.aliases("labels", "description"), vat.description
        )
        self.session.select_semantic_option(
            self.profile.aliases("labels", "vat_code"),
            self.profile.aliases("choices", "vat_standard_rate"),
        )
        self.session.set_labeled_decimal(
            self.profile.aliases("labels", "value"), vat.value
        )
        self._save_once()
        return f"vat-{uuid4().hex}"

    def search_products(self, sku: str) -> list[ProductCandidate]:
        self._activate_order_editor()
        self._order_row_ids_before_selector = {
            _runtime_id(row)
            for row in self.session.data_rows(scope=self.session.active_window())
        }
        self.session.invoke_related_image(
            self.profile.aliases("labels", "items"), ordinal=0
        )
        dialog = self.session.wait_for_named_window(
            self.profile.aliases("titles", "select_product"),
            timeout=self.profile.editor_timeout_seconds,
        )
        self._selector_dialog = dialog
        self.session.set_search_query(
            self.profile.aliases("labels", "search"), sku, scope=dialog
        )
        rows = self.session.wait_for_stable_data_rows(
            scope=dialog,
            stable_seconds=float(
                self.profile.values.get("search_stabilization_seconds", 0.5)
            ),
            timeout=self.profile.editor_timeout_seconds,
        )
        output = []
        for row in rows:
            candidate_sku = self.session.row_value_for_header(
                row, self.profile.aliases("columns", "item_number"), scope=dialog
            )
            if normalize_text(candidate_sku) != normalize_text(sku):
                continue
            record_id = _runtime_id(row)
            candidate = ProductCandidate(
                record_id=record_id,
                sku=candidate_sku,
                name=self.session.row_value_for_header(
                    row, self.profile.aliases("columns", "product_name"), scope=dialog
                ),
                vat_percent=Decimal("0"),
                gross_price=Decimal("0"),
            )
            self._records[record_id] = row
            self._product_skus[record_id] = candidate_sku
            output.append(candidate)
        return output

    def create_product(self, product: ProductCandidate) -> str:
        existing = self._tab_runtime_ids(self.profile.aliases("titles", "new_product"))
        if existing:
            raise ManualReviewRequired(
                "Fakturama already has an unsaved 'New Product' editor open. "
                "Close that tab (discard it only if it is safe), then rerun; "
                "automation refuses to overwrite or reuse an existing draft."
            )
        self.session.invoke_text(self.profile.aliases("actions", "new_product"))
        self.session.wait_for_control(
            self.profile.aliases("titles", "new_product"),
            control_types=("TabItem",),
            scope=self.session.root(),
            exclude_runtime_ids=existing,
            timeout=self.profile.editor_timeout_seconds,
        )
        for key, value in (
            ("item_number", product.sku),
            ("name", product.name),
            ("description", product.name),
        ):
            self.session.set_labeled_value(self.profile.aliases("labels", key), value)
        for key, value in (
            ("gross_price", product.gross_price),
            ("cost_price", Decimal("0.00")),
            ("stock", Decimal("0.00")),
        ):
            self.session.set_labeled_decimal(self.profile.aliases("labels", key), value)
        self.session.select_semantic_option(
            self.profile.aliases("labels", "product_vat"),
            [f"VAT {format(product.vat_percent.normalize(), 'f')}%"],
        )
        self._save_once()
        return f"product-{uuid4().hex}"

    def add_product_to_order(self, record_id: str) -> int:
        row = self._records.get(record_id)
        if row is None:
            raise KeyError(record_id)
        try:
            row.select()
        except Exception:
            row.click_input()
        dialog = (
            self._selector_dialog
            if self._selector_dialog is not None
            else self.session.active_window()
        )
        self.session.invoke(self.profile.aliases("actions", "ok"), scope=dialog)
        self.session.wait_until_closed(
            dialog, timeout=self.profile.editor_timeout_seconds
        )
        self._selector_dialog = None
        expected_sku = self._product_skus.get(record_id)
        if not expected_sku:
            raise UnsupportedAutomation(
                f"selected Product {record_id!r} has no recorded exact SKU"
            )
        deadline = time.monotonic() + self.profile.editor_timeout_seconds
        matches = []
        while time.monotonic() < deadline:
            rows = self.session.data_rows(scope=self.session.active_window())
            matches = [
                candidate
                for candidate in rows
                if _runtime_id(candidate) not in self._order_row_ids_before_selector
                and normalize_text(expected_sku)
                in normalize_text(" ".join(_row_values(candidate)))
            ]
            if len(matches) == 1:
                break
            if len(matches) > 1:
                raise UnsupportedAutomation(
                    f"multiple new Order rows contain SKU {expected_sku!r}"
                )
            time.sleep(0.1)
        if len(matches) != 1:
            raise TransientUiError(
                f"new Order row for SKU {expected_sku!r} did not appear"
            )
        self._order_rows.append(matches[0])
        return len(self._order_rows) - 1

    def set_order_line(self, row_index: int, item: OrderItem) -> None:
        row = self._order_rows[row_index]
        values = {
            "quantity": str(item.quantity),
            "unit_price": str(item.unit_net_price),
            "vat": str(item.vat_percent),
            "line_discount": str(item.discount_percent),
        }
        for key, value in values.items():
            self.session.set_grid_cell(
                row, self.profile.aliases("columns", key), value, scope=self.session.root()
            )
        expected_cells = {
            "quantity": item.quantity,
            "unit_price": item.unit_net_price,
            "vat": item.vat_percent,
            "line_discount": item.discount_percent,
            "line_price": item.source_total,
        }
        for key, expected in expected_cells.items():
            raw = self.session.row_value_for_header(
                row,
                self.profile.aliases("columns", key),
                scope=self.session.active_window(),
            )
            actual = _parse_decimal(raw)
            if abs(actual - expected) > Decimal("0.01"):
                raise VerificationError(
                    f"Order line {key} is {actual}, expected {expected}"
                )
        self._assert_row_contains(
            row,
            (
                item.sku,
                str(item.quantity),
                _decimal_ui(item.unit_net_price),
                str(item.vat_percent),
                str(item.discount_percent),
                _decimal_ui(item.source_total),
            ),
        )

    def complete_order(self) -> None:
        """Apply the document-level defaults at Take Home §4.2."""
        self._activate_order_editor()
        self.session.set_labeled_decimal(
            self.profile.aliases("labels", "order_discount"), Decimal("0")
        )
        self.session.select_semantic_option(
            self.profile.aliases("labels", "shipping"),
            self.profile.aliases("choices", "free_shipping"),
        )

    def read_order_snapshot(self) -> OrderSnapshot:
        order = self._require_order()
        self._activate_order_editor()
        active = self.session.active_window()
        actual_number = self.session.read_labeled_value(
            self.profile.aliases("labels", "document_number"), scope=active
        )
        if normalize_text(actual_number) != normalize_text(self.order_number):
            raise VerificationError(
                f"Order No. changed from {self.order_number!r} to {actual_number!r}"
            )
        actual_date = self._read_date("date", scope=active)
        external_reference = self.session.read_labeled_value(
            self.profile.aliases("labels", "external_reference"), scope=active
        )
        if normalize_text(external_reference) != normalize_text(order.external_reference):
            raise VerificationError(
                f"Order Cust.Ref. is {external_reference!r}, expected "
                f"{order.external_reference!r}"
            )
        price_mode = self.session.read_semantic_option(
            self.profile.aliases("labels", "order_price_mode"),
            identifying_options=(
                self.profile.aliases("choices", "net")
                + self.profile.aliases("choices", "gross")
            ),
            scope=active,
        )
        self._assert_choice(price_mode, "net", "Order price mode")
        vat_mode = self.session.read_semantic_option(
            self.profile.aliases("labels", "order_vat_mode"),
            identifying_options=(
                self.profile.aliases("choices", "with_vat")
                + self.profile.aliases("choices", "without_vat")
            ),
            scope=active,
        )
        self._assert_choice(vat_mode, "with_vat", "Order VAT mode")
        overall_discount = _parse_decimal(
            self.session.read_labeled_value(
                self.profile.aliases("labels", "order_discount"), scope=active
            )
        )
        if overall_discount != Decimal("0"):
            raise VerificationError(
                f"Order overall Discount is {overall_discount}, expected 0"
            )
        shipping = self.session.read_semantic_option(
            self.profile.aliases("labels", "shipping"),
            identifying_options=self.profile.aliases("choices", "free_shipping"),
            scope=active,
        )
        self._assert_choice(shipping, "free_shipping", "Order Shipping")
        if len(self._order_rows) != len(order.items):
            raise VerificationError("tracked Order row count differs from source")
        actual_items = []
        for row, item in zip(self._order_rows, order.items, strict=True):
            self._assert_row_contains(
                row,
                (
                    item.sku,
                    str(item.quantity),
                    _decimal_ui(item.unit_net_price),
                    str(item.vat_percent),
                    str(item.discount_percent),
                    _decimal_ui(item.source_total),
                ),
            )
            actual_items.append(
                OrderItem(
                    sku=self.session.row_value_for_header(
                        row,
                        self.profile.aliases("columns", "item_number"),
                        scope=active,
                    ),
                    description=item.description,
                    quantity=_parse_decimal(
                        self.session.row_value_for_header(
                            row,
                            self.profile.aliases("columns", "quantity"),
                            scope=active,
                        )
                    ),
                    unit_net_price=_parse_decimal(
                        self.session.row_value_for_header(
                            row,
                            self.profile.aliases("columns", "unit_price"),
                            scope=active,
                        )
                    ),
                    vat_percent=_parse_decimal(
                        self.session.row_value_for_header(
                            row,
                            self.profile.aliases("columns", "vat"),
                            scope=active,
                        )
                    ),
                    discount_percent=_parse_decimal(
                        self.session.row_value_for_header(
                            row,
                            self.profile.aliases("columns", "line_discount"),
                            scope=active,
                        )
                    ),
                    source_total=_parse_decimal(
                        self.session.row_value_for_header(
                            row,
                            self.profile.aliases("columns", "line_price"),
                            scope=active,
                        )
                    ),
                )
            )
        totals = self._read_totals_or_expected(order)
        return OrderSnapshot(
            number=actual_number,
            order_date=actual_date,
            external_reference=external_reference,
            debtor=self.read_selected_debtor(),
            # The specification verifies this field on the linked Invoice,
            # not as an Order-header field.
            payment_method=order.payment.method,
            items=actual_items,
            totals=totals,
            state="open",
        )

    def save_order(self) -> DocumentRecord:
        snapshot = self.read_order_snapshot()
        self._save_once()
        document = DocumentRecord(
            record_id=f"order-{snapshot.number}",
            document_type="Order",
            number=snapshot.number,
            date=snapshot.order_date,
            external_reference=snapshot.external_reference,
            state="open",
            total=snapshot.totals.total_gross,
            transaction_id=self.transaction_id,
        )
        self.order_document = document
        return document

    def verify_document(self, expected: DocumentRecord) -> DocumentRecord:
        self._open_data_view("documents")
        matching_rows = [
            row
            for row in self.session.data_rows(scope=self.session.root())
            if normalize_text(expected.number)
            in normalize_text(" ".join(_row_values(row)))
        ]
        if len(matching_rows) != 1:
            raise VerificationError(
                f"expected one persisted row for {expected.number!r}, "
                f"found {len(matching_rows)}"
            )
        row = matching_rows[0]
        self._assert_row_contains(
            row,
            (
                expected.document_type,
                expected.number,
                expected.external_reference,
                _decimal_ui(expected.total),
                expected.state,
            ),
        )
        row_text = normalize_text(" ".join(_row_values(row)))
        date_variants = {
            expected.date.strftime(date_format)
            for date_format in (
                self.profile.date_format,
                "%d.%m.%Y",
                "%m/%d/%Y",
                "%d/%m/%Y",
            )
        }
        if not any(normalize_text(value) in row_text for value in date_variants):
            raise VerificationError(
                f"persisted document row does not contain expected date "
                f"{expected.date.isoformat()!r}"
            )
        return expected

    def create_follow_up_invoice(self) -> str:
        if self.order_document is None:
            raise RuntimeError("Order must be saved first")
        self._activate_order_editor()
        existing = self._tab_runtime_ids(self.profile.aliases("titles", "new_invoice"))
        if existing:
            raise ManualReviewRequired(
                "Fakturama already has an unsaved 'New Invoice' editor open. "
                "Close that tab (discard it only if it is safe), then rerun; "
                "automation refuses to overwrite or reuse an existing draft."
            )
        self.session.invoke_in_group(
            self.profile.aliases("groups", "follow_up"),
            self.profile.aliases("actions", "follow_up_invoice"),
            scope=self.session.active_window(),
        )
        tab = self.session.wait_for_control(
            self.profile.aliases("titles", "new_invoice"),
            control_types=("TabItem",),
            scope=self.session.root(),
            exclude_runtime_ids=existing,
            timeout=self.profile.editor_timeout_seconds,
        )
        self._invoice_editor_tab_id = _runtime_id(tab)
        self.invoice_number = self.session.read_labeled_value(
            self.profile.aliases("labels", "document_number")
        )
        return self.invoice_number

    def read_invoice_snapshot(self) -> InvoiceSnapshot:
        order = self._require_order()
        self._activate_invoice_editor()
        active = self.session.active_window()
        actual_number = self.session.read_labeled_value(
            self.profile.aliases("labels", "document_number"), scope=active
        )
        if normalize_text(actual_number) != normalize_text(self.invoice_number):
            raise VerificationError(
                f"Invoice No. changed from {self.invoice_number!r} to {actual_number!r}"
            )
        invoice_date = self._read_date("date", scope=active)
        service_date = self._read_date("service_date", scope=active)
        order_date = self._read_date("order_date", scope=active)
        external_reference = self.session.read_labeled_value(
            self.profile.aliases("labels", "external_reference"), scope=active
        )
        self._assert_labeled_address(
            "invoice_address", order.debtor.billing_address, order.debtor.company
        )
        if not has_distinct_delivery_address(order.debtor):
            self._assert_labeled_address(
                "delivery_address",
                order.debtor.effective_delivery_address,
                order.debtor.company,
            )
        vat_mode = self.session.read_semantic_option(
            self.profile.aliases("labels", "order_vat_mode"),
            identifying_options=(
                self.profile.aliases("choices", "with_vat")
                + self.profile.aliases("choices", "without_vat")
            ),
            scope=active,
        )
        self._assert_choice(vat_mode, "with_vat", "Invoice VAT mode")
        payment_method = self.session.read_semantic_option(
            self.profile.aliases("labels", "payment_method"),
            identifying_options=[order.payment.method],
            scope=active,
        )
        self._assert_item_table(order, scope=active)
        totals = self._read_totals_or_expected(order)
        snapshot = InvoiceSnapshot(
            number=actual_number,
            invoice_date=invoice_date,
            service_date=service_date,
            order_date=order_date,
            external_reference=external_reference,
            debtor=self.read_selected_debtor(),
            payment=Payment(method=payment_method, status=PaymentStatus.UNPAID),
            items=order.items,
            totals=totals,
            state="unpaid",
        )
        self._last_invoice_snapshot = snapshot
        return snapshot

    def set_invoice_payment(self, payment: Payment, invoice_total: Decimal) -> None:
        self._activate_invoice_editor()
        if not self.session.select_optional_semantic_option(
            self.profile.aliases("labels", "payment_method"), [payment.method]
        ):
            raise ManualReviewRequired(
                f"Invoice Payment Method {payment.method!r} is not available"
            )
        if payment.status is PaymentStatus.PAID:
            self.session.set_toggle(self.profile.aliases("actions", "paid"), True)
            self.session.set_labeled_date(
                self.profile.aliases("labels", "payment_date"),
                payment.payment_date,
                date_format=self.profile.date_format,
            )
            self.session.set_labeled_decimal(
                self.profile.aliases("labels", "payment_value"),
                invoice_total,
            )
        else:
            self.session.set_toggle(self.profile.aliases("actions", "paid"), False)
        self.invoice_payment = payment

    def save_invoice(self) -> DocumentRecord:
        order = self._require_order()
        if self.invoice_number is None or self.invoice_payment is None:
            raise RuntimeError("Invoice payment stage is incomplete")
        self._activate_invoice_editor()
        actual_number = self.session.read_labeled_value(
            self.profile.aliases("labels", "document_number")
        )
        if normalize_text(actual_number) != normalize_text(self.invoice_number):
            raise VerificationError(
                f"Invoice No. changed from {self.invoice_number!r} to {actual_number!r}"
            )
        self._save_once()
        state = "paid" if self.invoice_payment.status is PaymentStatus.PAID else "unpaid"
        invoice_date = (
            self._last_invoice_snapshot.invoice_date
            if self._last_invoice_snapshot is not None
            else self._read_date("date")
        )
        return DocumentRecord(
            record_id=f"invoice-{self.invoice_number}",
            document_type="Invoice",
            number=actual_number,
            date=invoice_date,
            external_reference=order.external_reference,
            state=state,
            total=order.totals.total_gross,
            transaction_id=self.transaction_id,
        )

    def capture_screenshot(self, path: Path) -> bool:
        if self.session.main is None:
            return False
        image = self.session.root().capture_as_image()
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path)
        return True

    def _open_data_view(self, key: str) -> None:
        aliases = self.profile.aliases("data_views", key)
        try:
            self.session.root().menu_select("->".join(aliases))
        except Exception as exc:
            raise UnsupportedAutomation(
                f"could not open the documented menu path {' > '.join(aliases)!r}"
            ) from exc

    def _activate_editor(self, number: str | None) -> None:
        if not number:
            return
        matches = []
        for tab in self.session.root().descendants(control_type="TabItem"):
            try:
                if normalize_text(number) in normalize_text(tab.window_text()):
                    matches.append(tab)
            except Exception:
                continue
        if len(matches) != 1:
            raise UnsupportedAutomation(
                f"could not identify the open editor tab for document {number!r}"
            )
        try:
            matches[0].select()
        except Exception:
            matches[0].click_input()

    def _activate_order_editor(self) -> None:
        if self._order_editor_tab_id:
            fallback_names = self.profile.aliases("titles", "new_order")
            if self.order_number:
                fallback_names = [*fallback_names, self.order_number]
            self._order_editor_tab_id = self._activate_editor_tab(
                self._order_editor_tab_id,
                fallback_names=fallback_names,
            )
            return
        self._activate_editor(self.order_number)

    def _activate_invoice_editor(self) -> None:
        if not self._invoice_editor_tab_id:
            raise UnsupportedAutomation("linked Invoice editor tab was not recorded")
        fallback_names = self.profile.aliases("titles", "new_invoice")
        if self.invoice_number:
            fallback_names = [*fallback_names, self.invoice_number]
        self._invoice_editor_tab_id = self._activate_editor_tab(
            self._invoice_editor_tab_id,
            fallback_names=fallback_names,
        )

    def _activate_editor_tab(
        self,
        runtime_id: str | None,
        *,
        fallback_names: list[str] | None = None,
    ) -> str:
        """Activate an editor, reacquiring it when SWT replaces its runtime ID."""
        matches = [
            tab
            for tab in self.session.root().descendants(control_type="TabItem")
            if _runtime_id(tab) == runtime_id
        ]
        if len(matches) != 1 and fallback_names:
            expected = [
                normalize_text(name)
                for name in fallback_names
                if name and normalize_text(name)
            ]
            all_tabs = list(self.session.root().descendants(control_type="TabItem"))
            exact_matches = []
            for tab in all_tabs:
                try:
                    actual = normalize_text(tab.window_text()).lstrip("* ").strip()
                    if actual in expected:
                        exact_matches.append(tab)
                except Exception:
                    continue
            if len(exact_matches) == 1:
                matches = exact_matches
            elif not exact_matches:
                sub_matches = []
                for tab in all_tabs:
                    try:
                        actual = normalize_text(tab.window_text()).lstrip("* ").strip()
                        if actual and any(
                            name in actual or actual in name
                            for name in expected
                            if len(name) >= 3
                        ):
                            sub_matches.append(tab)
                    except Exception:
                        continue
                if len(sub_matches) == 1:
                    matches = sub_matches
                elif len(sub_matches) > 1:
                    dirty = [
                        t
                        for t in sub_matches
                        if "*" in str(getattr(t, "window_text", lambda: "")())
                    ]
                    if len(dirty) == 1:
                        matches = dirty
                    else:
                        matches = sub_matches
                else:
                    matches = []
            else:
                matches = exact_matches
        if len(matches) != 1:
            raise UnsupportedAutomation(
                f"could not identify one editor tab from recorded ID {runtime_id!r} "
                f"or names {fallback_names!r}"
            )
        try:
            matches[0].select()
        except Exception:
            matches[0].click_input()
        return _runtime_id(matches[0])

    def _tab_runtime_ids(self, names: list[str]) -> set[str]:
        expected = {normalize_text(name) for name in names}
        return {
            _runtime_id(tab)
            for tab in self.session.root().descendants(control_type="TabItem")
            if normalize_text(tab.window_text()) in expected
        }

    def _activate_editor_by_names(self, names: list[str]) -> None:
        expected = [normalize_text(name) for name in names]
        matches = []
        for tab in self.session.root().descendants(control_type="TabItem"):
            try:
                actual = normalize_text(tab.window_text())
                if actual and any(name in actual for name in expected):
                    matches.append(tab)
            except Exception:
                continue
        if len(matches) != 1:
            raise UnsupportedAutomation(
                f"could not identify one open editor tab matching {names!r}"
            )
        try:
            matches[0].select()
        except Exception:
            matches[0].click_input()

    def _search_simple_rows(self, query: str, model, kind: str):
        self.session.set_search_query(
            self.profile.aliases("labels", "search"), query
        )
        rows = self.session.wait_for_stable_data_rows(
            scope=self.session.active_window(),
            stable_seconds=float(
                self.profile.values.get("search_stabilization_seconds", 0.5)
            ),
            timeout=self.profile.editor_timeout_seconds,
        )
        output = []
        for row in rows:
            record_id = _runtime_id(row)
            if kind == "payment":
                name = self.session.row_value_for_header(
                    row, self.profile.aliases("columns", "payment_name")
                )
                if normalize_text(name) != normalize_text(query):
                    continue
                item = model(
                    record_id=record_id,
                    name=name,
                    description=self.session.row_value_for_header(
                        row, self.profile.aliases("columns", "description")
                    ),
                    payment_code=self.session.row_value_for_header(
                        row, self.profile.aliases("columns", "payment_code")
                    ),
                )
            else:
                name = self.session.row_value_for_header(
                    row, self.profile.aliases("columns", "vat_name")
                )
                if normalize_text(name) != normalize_text(query):
                    continue
                raw_value = self.session.row_value_for_header(
                    row, self.profile.aliases("columns", "vat_value")
                )
                number = _parse_decimal(raw_value)
                raw_code = self.session.row_value_for_header(
                    row, self.profile.aliases("columns", "vat_code")
                )
                code = "S" if normalize_text(raw_code).startswith("s") else raw_code
                item = model(
                    record_id=record_id,
                    name=name,
                    description=self.session.row_value_for_header(
                        row, self.profile.aliases("columns", "description")
                    ),
                    value=number,
                    e_invoice_code=code,
                )
            self._records[record_id] = row
            output.append(item)
        return output

    def _debtor_candidate_from_row(self, row, scope):
        try:
            record_id = _runtime_id(row)
            candidate = DebtorCandidate(
                record_id=record_id,
                company=self.session.row_value_for_header(
                    row,
                    self.profile.aliases("columns", "company"),
                    scope=scope,
                    allow_empty=True,
                ),
                first_name=self.session.row_value_for_header(
                    row,
                    self.profile.aliases("columns", "first_name"),
                    scope=scope,
                    allow_empty=True,
                ),
                last_name=self.session.row_value_for_header(
                    row,
                    self.profile.aliases("columns", "last_name"),
                    scope=scope,
                    allow_empty=True,
                ),
                billing_address=Address(
                    # The selector exposes only the fields required by §2.3.
                    # Full addresses are verified after the row is selected.
                    street="not exposed in selector",
                    zip=self.session.row_value_for_header(
                        row, self.profile.aliases("columns", "zip"), scope=scope
                    ),
                    city=self.session.row_value_for_header(
                        row, self.profile.aliases("columns", "city"), scope=scope
                    ),
                    country="not exposed in selector",
                ),
            )
        except ValueError:
            return None
        self._records[record_id] = row
        return candidate

    def _fill_address(self, address: Address) -> None:
        self.session.select_tab(self.profile.aliases("tabs", "addresses"))
        # Fill the portion currently visible in Main address first. ZIP, City,
        # and Country are below the fold in Fakturama's Debtor editor.
        for key, value in (
            ("street", address.street),
            ("email", address.email),
            ("telephone", address.telephone),
            ("additional_name", address.additional_name),
            ("address_specification", address.address_specification),
            ("district", address.district),
        ):
            if value:
                self.session.set_labeled_value(self.profile.aliases("labels", key), value)

        self.session.scroll_until_label(
            self.profile.aliases("labels", "zip"),
            anchor_names=self.profile.aliases("labels", "street"),
            direction="down",
            exact=False,
        )
        self.session.set_labeled_row_values(
            self.profile.aliases("labels", "zip"),
            [address.zip, address.city],
            exact=False,
        )
        if address.country:
            self.session.select_combo_by_initial(
                self.profile.aliases("labels", "country"), address.country
            )

    def _expand_address_roles(self) -> None:
        self.session.expand_labeled_control(
            self.profile.aliases("labels", "address_type")
        )
        self.session.wait_for_control(
            self.profile.aliases("actions", "invoice_address_role"),
            control_types=("CheckBox", "Button"),
            scope=self.session.root(),
            timeout=self.profile.editor_timeout_seconds,
        )
    def _assert_labeled_address(
        self, label_key: str, address: Address, company: str
    ) -> None:
        value = self.session.read_tab_value(
            self.profile.aliases("labels", label_key)
        )
        normalized = normalize_text(value)
        for expected in (
            company,
            address.additional_name,
            address.street,
            address.zip,
            address.city,
            address.country,
        ):
            if expected and normalize_text(expected) not in normalized:
                raise VerificationError(
                    f"{label_key} does not contain expected value {expected!r}"
                )

    def _assert_visible_text(self, value: str) -> None:
        expected = normalize_text(value)
        visible = []
        for element in self.session.root().descendants():
            try:
                if element.is_visible():
                    text = element.window_text()
                    if text:
                        visible.append(normalize_text(text))
            except Exception:
                continue
        if not any(expected in item for item in visible):
            raise VerificationError(f"expected visible text {value!r} was not found")

    def _read_totals_or_expected(self, order: OrderInput):
        values = {}
        for key, expected in (
            ("total_net", order.totals.total_net),
            ("total_vat", order.totals.total_vat),
            ("total_gross", order.totals.total_gross),
        ):
            raw = self.session.read_labeled_value(self.profile.aliases("labels", key))
            actual = _parse_decimal(raw)
            if abs(actual - expected) > Decimal("0.01"):
                raise VerificationError(
                    f"displayed {key} {actual} differs from source {expected}"
                )
            values[key] = actual
        return OrderTotals(**values)

    def _read_date(self, label_key: str, *, scope=None) -> date:
        raw = self.session.read_labeled_value(
            self.profile.aliases("labels", label_key), scope=scope
        ).strip()
        formats = (
            self.profile.date_format,
            "%d.%m.%Y",
            "%m/%d/%Y",
            "%d/%m/%Y",
            "%b %d, %Y",
        )
        for date_format in dict.fromkeys(formats):
            try:
                return datetime.strptime(raw, date_format).date()
            except ValueError:
                continue
        raise VerificationError(
            f"field {label_key!r} contains unsupported date value {raw!r}"
        )

    def _assert_choice(self, actual: str, choice_key: str, field_name: str) -> None:
        expected = {
            normalize_text(value) for value in self.profile.aliases("choices", choice_key)
        }
        if normalize_text(actual) not in expected:
            raise VerificationError(
                f"{field_name} is {actual!r}, expected one of "
                f"{self.profile.aliases('choices', choice_key)!r}"
            )

    def _assert_item_table(self, order: OrderInput, *, scope) -> None:
        rows = self.session.data_rows(scope=scope)
        remaining = list(rows)
        for item in order.items:
            matches = [
                row
                for row in remaining
                if normalize_text(item.sku)
                in normalize_text(" ".join(_row_values(row)))
            ]
            if not matches:
                raise VerificationError(
                    f"Invoice has no remaining item row for SKU {item.sku!r}"
                )
            matches.sort(key=lambda candidate: candidate.rectangle().top)
            row = matches[0]
            remaining.remove(row)
            self._assert_row_contains(
                row,
                (
                    item.sku,
                    str(item.quantity),
                    _decimal_ui(item.unit_net_price),
                    str(item.vat_percent),
                    str(item.discount_percent),
                    _decimal_ui(item.source_total),
                ),
            )

    def _assert_row_contains(self, row, expected_values) -> None:
        row_text = normalize_text(" ".join(_row_values(row)))
        numeric_row_text = _numeric_text(row_text)
        for expected in expected_values:
            text = str(expected)
            normalized = normalize_text(text)
            if normalized in row_text:
                continue
            numeric = _numeric_text(normalized)
            if numeric and numeric in numeric_row_text:
                continue
            raise VerificationError(f"table row does not contain {text!r}: {row_text!r}")

    def _save_once(self) -> None:
        try:
            self.session.invoke(
                self.profile.aliases("actions", "save"), scope=self.session.root()
            )
        except UnsupportedAutomation:
            saved = False
            for target in (
                lambda: self.session.active_window(),
                lambda: self.session.root(),
            ):
                try:
                    target().type_keys("^s")
                    saved = True
                    break
                except Exception:
                    continue
            if not saved:
                raise
        time.sleep(float(self.profile.values.get("save_stabilization_seconds", 0.5)))

    def _cancel_dialog(self, dialog) -> None:
        self.session.invoke(self.profile.aliases("actions", "cancel"), scope=dialog)
        self.session.wait_until_closed(
            dialog, timeout=self.profile.editor_timeout_seconds
        )

    def _require_order(self) -> OrderInput:
        if self.expected_order is None or self.order_number is None:
            raise RuntimeError("no expected Order is loaded")
        return self.expected_order


def inspect_uia_tree(executable: Path, profile_path: Path, output_path: Path) -> int:
    """Attach/start Fakturama and export a sanitized UIA control inventory."""
    profile = UiaProfile.load(profile_path)
    session = SemanticUiaSession(executable, profile)
    session.attach()
    rows = []
    for element in [session.root(), *session.root().descendants()]:
        try:
            info = element.element_info
            rectangle = element.rectangle()
            rows.append(
                {
                    "depth": _uia_depth(element, session.root()),
                    "control_type": info.control_type,
                    "name": element.window_text(),
                    "automation_id": info.automation_id,
                    "class_name": info.class_name,
                    "visible": element.is_visible(),
                    "enabled": element.is_enabled(),
                    "rectangle": [
                        rectangle.left,
                        rectangle.top,
                        rectangle.right,
                        rectangle.bottom,
                    ],
                }
            )
        except Exception:
            continue
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(rows)


def _uia_depth(element, root) -> int:
    depth = 0
    current = element
    while current != root and depth < 30:
        try:
            current = current.parent()
        except Exception:
            break
        depth += 1
    return depth


def _row_values(row) -> list[str]:
    values = []
    for item in [row, *row.descendants()]:
        try:
            text = item.window_text().strip()
        except Exception:
            continue
        if text and text not in values:
            values.append(text)
    return values


def _runtime_id(element) -> str:
    runtime_id = element.element_info.runtime_id
    return "uia-" + "-".join(str(value) for value in runtime_id)


def _first_decimal(values: list[str], default: Decimal) -> Decimal:
    for value in values:
        parsed = parse_decimal_text(value)
        if parsed is not None:
            return parsed
    return default


def _decimal_ui(value: Decimal) -> str:
    return f"{value:.2f}"


def _parse_decimal(value: str) -> Decimal:
    parsed = parse_decimal_text(value)
    if parsed is None:
        raise VerificationError(f"could not parse displayed decimal {value!r}")
    return parsed


def _numeric_text(value: str) -> str:
    return re.sub(r"[^0-9.-]", "", value.replace(",", "."))
