import json
from datetime import date
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

import fakturama_automation.gateways.uia as uia_module
from fakturama_automation.gateways.uia import (
    PywinautoFakturamaGateway,
    SemanticUiaSession,
    UiaProfile,
    _parse_decimal,
)
from fakturama_automation.domain.models import (
    Address,
    Debtor,
    PaymentMethodCandidate,
)


def test_uia_profile_is_valid_json() -> None:
    path = Path("config/fakturama-2.2.0-en.json")
    json.loads(path.read_text(encoding="utf-8"))
    profile = UiaProfile.load(path)

    assert "Order" in profile.aliases("actions", "new_order")
    assert profile.startup_timeout_seconds == 300
    assert profile.editor_timeout_seconds == 20
    assert profile.date_segment_order == ("month", "day", "year")
    assert "Search:" in profile.aliases("labels", "search")
    assert "First Name Last Name" in profile.aliases("labels", "debtor_name_row")
    assert profile.aliases("labels", "zip") == ["ZIP"]
    assert "address type" in profile.aliases("labels", "address_type")
    assert "add_address" not in profile.values["actions"]
    assert "add" not in profile.values["actions"]
    assert profile.aliases("columns", "line_price") == ["Price"]
    assert "Documents" in profile.startup_ready_names


def test_documented_list_plus_does_not_use_generic_add_name(tmp_path: Path) -> None:
    class Element:
        def __init__(self, text: str) -> None:
            self.text = text
            self.element_info = SimpleNamespace(control_type="Button")
            self.invoked = 0

        def window_text(self):
            return self.text

        def is_visible(self):
            return True

        def is_enabled(self):
            return True

        def invoke(self):
            self.invoked += 1

    plus = Element("+")
    generic_add = Element("Add")
    scope = SimpleNamespace(descendants=lambda: [generic_add, plus])
    session = SemanticUiaSession(
        tmp_path / "Fakturama.exe", UiaProfile({"window_title_re": ".*"})
    )

    session.invoke_upper_right_list_plus(scope=scope)

    assert plus.invoked == 1
    assert generic_add.invoked == 0


def test_unnamed_list_plus_is_grounded_to_table_upper_right(tmp_path: Path) -> None:
    class Rect:
        def __init__(self, left, top, right, bottom):
            self.left = left
            self.top = top
            self.right = right
            self.bottom = bottom

    class Element:
        def __init__(self, text, control_type, rect):
            self.text = text
            self.element_info = SimpleNamespace(control_type=control_type)
            self.rect = rect
            self.invoked = 0

        def window_text(self):
            return self.text

        def is_visible(self):
            return True

        def is_enabled(self):
            return True

        def rectangle(self):
            return self.rect

        def invoke(self):
            self.invoked += 1

    header = Element("Name", "HeaderItem", Rect(100, 100, 500, 120))
    intended = Element("", "Button", Rect(480, 40, 500, 60))
    distractor = Element("", "Button", Rect(20, 40, 40, 60))
    scope = SimpleNamespace(descendants=lambda: [header, distractor, intended])
    session = SemanticUiaSession(
        tmp_path / "Fakturama.exe", UiaProfile({"window_title_re": ".*"})
    )

    session.invoke_upper_right_list_plus(scope=scope)

    assert intended.invoked == 1
    assert distractor.invoked == 0


def test_new_payment_method_action_uses_named_command_before_geometry_fallback(
    tmp_path: Path,
) -> None:
    class FakeSession:
        def __init__(self):
            self.calls = []

        def invoke(self, names):
            self.calls.append(("invoke", names))

        def invoke_upper_right_list_plus(self, *, scope=None):
            self.calls.append(("plus", scope))

        def set_labeled_value(self, label_names, value):
            self.calls.append(("set_labeled_value", label_names, value))

        def set_labeled_decimal(self, label_names, value):
            self.calls.append(("set_labeled_decimal", label_names, str(value)))

        def select_semantic_option(self, label_names, option_names, **kwargs):
            self.calls.append(("select_semantic_option", label_names, option_names, kwargs))

        def active_window(self):
            return None

    gateway = object.__new__(PywinautoFakturamaGateway)
    session = FakeSession()
    gateway.session = session
    gateway.profile = UiaProfile.load(Path("config/fakturama-2.2.0-en.json"))
    gateway._save_once = lambda: None

    payment = PaymentMethodCandidate(
        record_id="payment-1",
        name="Bank Transfer",
        description="Bank transfer",
        payment_code="T",
    )
    gateway.session = session
    gateway.create_payment_method(payment)

    assert session.calls[0] == (
        "invoke",
        ["Create a new term of payment", "Create a new payment method", "New term of payment", "New payment method"],
    )


def test_combo_items_are_read_from_visible_popup_list_when_not_attached_to_combo() -> None:
    class PopupText:
        def __init__(self, text):
            self._text = text
            self.element_info = SimpleNamespace(control_type="Text")

        def window_text(self):
            return self._text

        def is_visible(self):
            return True

        def rectangle(self):
            return SimpleNamespace(left=10, top=20, right=200, bottom=30)

    class ComboBox:
        def __init__(self):
            self.element_info = SimpleNamespace(control_type="ComboBox")
            self._parent = SimpleNamespace(descendants=lambda: [self, PopupText("Credit transfer")])

        def item_texts(self):
            return []

        def children(self):
            return []

        def parent(self):
            return self._parent

        def expand(self):
            return None

        def legacy_properties(self):
            return {}

    combo = ComboBox()

    assert SemanticUiaSession._combo_item_texts(combo) == ["Credit transfer"]


def test_displayed_decimal_parsing_handles_common_locales() -> None:
    assert _parse_decimal("€1,234.56") == Decimal("1234.56")
    assert _parse_decimal("1.234,56 €") == Decimal("1234.56")
    assert _parse_decimal("267,70") == Decimal("267.70")


def test_payment_method_selection_does_not_require_payment_tab(tmp_path: Path) -> None:
    class FakeSession:
        def select_tab(self, names):
            raise uia_module.UnsupportedAutomation(
                f"expected one accessible control named {names!r}, found 0"
            )

        def select_optional_semantic_option(self, label_names, option_names):
            return (
                label_names == ["Payment", "Payment Method"]
                and option_names == ["Bank Transfer"]
            ) or (
                label_names == ["Payment"] and option_names == ["Bank Transfer"]
            )

    gateway = object.__new__(PywinautoFakturamaGateway)
    gateway._pending_debtor = object()
    gateway._debtor_editor_tab_id = "tab-1"
    gateway.session = FakeSession()
    gateway.profile = UiaProfile.load(Path("config/fakturama-2.2.0-en.json"))
    gateway._activate_editor_tab = lambda *_args, **_kwargs: "tab-1"

    assert gateway.select_debtor_payment_method("Bank Transfer") is True


def test_startup_discovers_ready_window_outside_launcher_process(
    tmp_path: Path, monkeypatch
) -> None:
    calls = []

    class Element:
        handle = 202

        def __init__(self, text: str) -> None:
            self.text = text

        def descendants(self):
            return [Element("Order")]

        def window_text(self):
            return self.text

        def is_visible(self):
            return True

        def is_enabled(self):
            return True

        def process_id(self):
            return 2222

    ready_window = Element("Fakturama 2")

    class FakeDesktop:
        def __init__(self):
            self.searches = 0

        def windows(self, **kwargs):
            calls.append(kwargs)
            self.searches += 1
            return [] if self.searches == 1 else [ready_window]

    desktop = FakeDesktop()

    class FakeApplication:
        connected_handle = None

        def __init__(self, **kwargs):
            pass

        def start(self, executable, wait_for_idle):
            return SimpleNamespace(process=1111)

        def connect(self, *, handle):
            FakeApplication.connected_handle = handle
            return self

        def window(self, *, handle):
            assert handle == ready_window.handle
            return ready_window

    monkeypatch.setattr(
        uia_module,
        "_load_pywinauto",
        lambda: (FakeApplication, lambda **kwargs: desktop, None),
    )
    profile = UiaProfile(
        {
            "window_title_re": ".*Fakturama.*",
            "startup_timeout_seconds": 1,
            "startup_poll_seconds": 0,
            "startup_ready_names": ["Order"],
        }
    )
    session = SemanticUiaSession(tmp_path / "Fakturama.exe", profile)

    session.attach()

    assert session.main is ready_window
    assert FakeApplication.connected_handle == 202
    assert all("process" not in search for search in calls)
    assert all(search.get("title_re") == ".*Fakturama.*" for search in calls)


def test_semantic_combo_fallback_uses_options_when_swt_hides_label(
    tmp_path: Path, monkeypatch
) -> None:
    class Combo:
        element_info = SimpleNamespace(control_type="ComboBox")

        def __init__(self, items):
            self.items = items
            self.selected = ""

        def item_texts(self):
            return self.items

        def is_visible(self):
            return True

        def is_enabled(self):
            return True

        def select(self, value):
            self.selected = value

        def get_value(self):
            return ""

        def selected_text(self):
            return self.selected

    net_combo = Combo(["Net", "Gross"])
    vat_combo = Combo(["With VAT", "Without VAT"])
    scope = SimpleNamespace(descendants=lambda: [net_combo, vat_combo])
    session = SemanticUiaSession(
        tmp_path / "Fakturama.exe", UiaProfile({"window_title_re": ".*"})
    )
    monkeypatch.setattr(
        session,
        "input_for_label",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            uia_module.UnsupportedAutomation("hidden SWT label")
        ),
    )

    session.select_semantic_option(
        ["Net or Gross"], ["Net"], distinguishing_options=["Gross"], scope=scope
    )

    assert net_combo.selected == "Net"
    assert vat_combo.selected == ""


def test_country_is_selected_by_initial_then_exact_option(
    tmp_path: Path, monkeypatch
) -> None:
    class CountryCombo:
        element_info = SimpleNamespace(control_type="ComboBox")

        def __init__(self):
            self.value = "United States"
            self.keys = []
            self.clicked = False
            self.selected = []

        def click_input(self):
            self.clicked = True

        def type_keys(self, value):
            self.keys.append(value)

        def item_texts(self):
            return ["France", "Germany", "United States"]

        def select(self, value):
            self.selected.append(value)
            self.value = value

        def get_value(self):
            return self.value

    combo = CountryCombo()
    session = SemanticUiaSession(
        tmp_path / "Fakturama.exe", UiaProfile({"window_title_re": ".*"})
    )
    monkeypatch.setattr(session, "input_for_label", lambda *args, **kwargs: combo)

    session.select_combo_by_initial(["Country"], "Germany")

    assert combo.clicked is True
    assert combo.value == "Germany"
    assert combo.selected == ["Germany"]
    assert combo.keys == ["G", "{ENTER}"]


def test_address_role_expander_is_selected_relative_to_address_type_field(
    tmp_path: Path, monkeypatch
) -> None:
    class Rect:
        def __init__(self, left, top, right, bottom):
            self.left, self.top, self.right, self.bottom = left, top, right, bottom

    class Element:
        def __init__(self, control_type, rect):
            self.element_info = SimpleNamespace(control_type=control_type)
            self.rect = rect
            self.invoked = 0

        def rectangle(self):
            return self.rect

        def is_visible(self):
            return True

        def is_enabled(self):
            return True

        def invoke(self):
            self.invoked += 1

    field = Element("Edit", Rect(170, 300, 590, 325))
    arrow = Element("Button", Rect(590, 300, 610, 325))
    unrelated_add = Element("Button", Rect(1100, 100, 1130, 130))
    scope = SimpleNamespace(descendants=lambda: [unrelated_add, arrow, field])
    session = SemanticUiaSession(
        tmp_path / "Fakturama.exe", UiaProfile({"window_title_re": ".*"})
    )
    monkeypatch.setattr(session, "input_for_label", lambda *args, **kwargs: field)

    session.expand_labeled_control(["address type"], scope=scope)

    assert arrow.invoked == 1
    assert unrelated_add.invoked == 0


def test_explicit_identical_delivery_assigns_both_roles_without_add_action(
    tmp_path: Path, monkeypatch
) -> None:
    gateway = PywinautoFakturamaGateway(
        tmp_path / "Fakturama.exe", Path("config/fakturama-2.2.0-en.json")
    )
    session = Mock()
    session.wait_for_control.return_value = SimpleNamespace(
        element_info=SimpleNamespace(runtime_id=(42, 7, 4, -17))
    )
    session.read_semantic_option.return_value = "---"
    gateway.session = session
    monkeypatch.setattr(gateway, "_tab_runtime_ids", lambda names: set())
    monkeypatch.setattr(gateway, "_fill_address", Mock())
    monkeypatch.setattr(gateway, "_expand_address_roles", Mock())
    address = Address(
        street="Example Street 12",
        zip="10115",
        city="Berlin",
        country="Germany",
    )
    debtor = Debtor(
        company="Example GmbH",
        first_name="Ada",
        last_name="Lovelace",
        billing_address=address,
        delivery_address=address.model_copy(),
    )

    gateway.open_new_debtor(debtor)

    assert session.set_toggle.call_args_list == [
        call(["Invoice address"], True),
        call(["Delivery address"], True),
    ]
    session.invoke_upper_right_list_plus.assert_not_called()


def test_distinct_delivery_assigns_only_main_invoice_role(
    tmp_path: Path, monkeypatch
) -> None:
    gateway = PywinautoFakturamaGateway(
        tmp_path / "Fakturama.exe", Path("config/fakturama-2.2.0-en.json")
    )
    session = Mock()
    session.wait_for_control.return_value = SimpleNamespace(
        element_info=SimpleNamespace(runtime_id=(42, 7, 4, -17))
    )
    session.read_semantic_option.return_value = "---"
    gateway.session = session
    monkeypatch.setattr(gateway, "_tab_runtime_ids", lambda names: set())
    monkeypatch.setattr(gateway, "_fill_address", Mock())
    monkeypatch.setattr(gateway, "_expand_address_roles", Mock())
    billing = Address(
        street="Friedrichstrasse 88",
        zip="10117",
        city="Berlin",
        country="Germany",
    )
    delivery = Address(
        street="Beusselstrasse 44",
        zip="10553",
        city="Berlin",
        country="Germany",
    )
    debtor = Debtor(
        company="Northstar Office GmbH",
        first_name="Marta",
        last_name="Klein",
        billing_address=billing,
        delivery_address=delivery,
    )

    gateway.open_new_debtor(debtor)

    assert session.set_toggle.call_args_list == [
        call(["Invoice address"], True),
    ]
    assert gateway._pending_debtor.company == "Northstar Office GmbH"
    assert gateway._pending_debtor.delivery_address is None


def test_unlabeled_modal_search_uses_the_only_editable_field(
    tmp_path: Path, monkeypatch
) -> None:
    class SearchField:
        element_info = SimpleNamespace(control_type="Edit")

        def __init__(self):
            self.value = ""
            self.clicked = False

        def is_visible(self):
            return True

        def is_enabled(self):
            return True

        def click_input(self):
            self.clicked = True

        def set_edit_text(self, value):
            self.value = value

        def get_value(self):
            return self.value

    field = SearchField()
    scope = SimpleNamespace(descendants=lambda: [field])
    session = SemanticUiaSession(
        tmp_path / "Fakturama.exe", UiaProfile({"window_title_re": ".*"})
    )
    monkeypatch.setattr(
        session,
        "input_for_label",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            uia_module.UnsupportedAutomation("Search label is hidden")
        ),
    )

    session.set_search_query(["Search", "Search:"], "Northwind GmbH", scope=scope)

    assert field.clicked is True
    assert field.value == "Northwind GmbH"


def test_unlabeled_modal_search_rejects_ambiguous_editable_fields(
    tmp_path: Path, monkeypatch
) -> None:
    class SearchField:
        element_info = SimpleNamespace(control_type="Edit")

        def is_visible(self):
            return True

        def is_enabled(self):
            return True

    scope = SimpleNamespace(descendants=lambda: [SearchField(), SearchField()])
    session = SemanticUiaSession(
        tmp_path / "Fakturama.exe", UiaProfile({"window_title_re": ".*"})
    )
    monkeypatch.setattr(
        session,
        "input_for_label",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            uia_module.UnsupportedAutomation("Search label is hidden")
        ),
    )

    with pytest.raises(uia_module.UnsupportedAutomation, match="2 editable fields"):
        session.set_search_query(["Search"], "query", scope=scope)


def test_named_modal_is_selected_instead_of_arbitrary_process_window(
    tmp_path: Path, monkeypatch
) -> None:
    class Window:
        def __init__(self, text, runtime_id):
            self.text = text
            self.element_info = SimpleNamespace(
                control_type="Window", runtime_id=runtime_id
            )

        def window_text(self):
            return self.text

    dialog = Window("Select the address", (42, 2))
    main = Window("Fakturama", (42, 1))
    desktop = SimpleNamespace(windows=lambda **kwargs: [dialog, main])
    root = SimpleNamespace(process_id=lambda: 42, descendants=lambda: [])
    session = SemanticUiaSession(
        tmp_path / "Fakturama.exe", UiaProfile({"window_title_re": ".*"})
    )
    session.main = root
    monkeypatch.setattr(
        uia_module,
        "_load_pywinauto",
        lambda: (None, lambda **kwargs: desktop, None),
    )

    found = session.wait_for_named_window(["Select the address"], timeout=0.1)

    assert found is dialog


def test_table_values_are_grounded_by_visible_header_geometry(tmp_path: Path) -> None:
    class Rect:
        def __init__(self, left, top, right, bottom):
            self.left, self.top, self.right, self.bottom = left, top, right, bottom

    class Element:
        def __init__(self, text, control_type, rect, children=None):
            self.text = text
            self.element_info = SimpleNamespace(control_type=control_type)
            self.rect = rect
            self.children = children or []

        def descendants(self, **kwargs):
            result = list(self.children)
            if "control_type" in kwargs:
                result = [
                    item
                    for item in result
                    if item.element_info.control_type == kwargs["control_type"]
                ]
            return result

        def rectangle(self):
            return self.rect

        def window_text(self):
            return self.text

        def is_visible(self):
            return True

        def is_enabled(self):
            return True

    company_header = Element("Company", "HeaderItem", Rect(0, 0, 100, 20))
    city_header = Element("City", "HeaderItem", Rect(100, 0, 200, 20))
    company_cell = Element("Acme GmbH", "Text", Rect(0, 20, 100, 40))
    city_cell = Element("Berlin", "Text", Rect(100, 20, 200, 40))
    row = Element("", "DataItem", Rect(0, 20, 200, 40), [company_cell, city_cell])
    scope = Element(
        "dialog", "Window", Rect(0, 0, 200, 100), [company_header, city_header]
    )
    session = SemanticUiaSession(
        tmp_path / "Fakturama.exe", UiaProfile({"window_title_re": ".*"})
    )

    assert session.row_value_for_header(row, ["Company"], scope=scope) == "Acme GmbH"
    assert session.row_value_for_header(row, ["City"], scope=scope) == "Berlin"


def test_combined_debtor_name_label_sets_two_adjacent_fields_only(
    tmp_path: Path,
) -> None:
    class Rect:
        def __init__(self, left, top, right, bottom):
            self.left, self.top, self.right, self.bottom = left, top, right, bottom

    class Element:
        def __init__(self, text, control_type, rect):
            self.text = text
            self.element_info = SimpleNamespace(control_type=control_type)
            self.rect = rect
            self.value = ""

        def window_text(self):
            return self.text

        def rectangle(self):
            return self.rect

        def is_visible(self):
            return True

        def is_enabled(self):
            return True

        def set_edit_text(self, value):
            self.value = value

        def get_value(self):
            return self.value

    label = Element("First Name Last Name", "Text", Rect(0, 50, 150, 70))
    first_name = Element("", "Edit", Rect(160, 50, 300, 70))
    last_name = Element("", "Edit", Rect(310, 50, 450, 70))
    customer_id = Element("", "Edit", Rect(160, 10, 300, 30))
    elements = [label, customer_id, last_name, first_name]
    scope = SimpleNamespace(
        descendants=lambda **kwargs: [
            element
            for element in elements
            if not kwargs
            or element.element_info.control_type == kwargs.get("control_type")
        ]
    )
    session = SemanticUiaSession(
        tmp_path / "Fakturama.exe", UiaProfile({"window_title_re": ".*"})
    )

    session.set_labeled_row_values(
        ["First Name Last Name"], ["Maya", "Hoffmann"], scope=scope
    )

    assert first_name.value == "Maya"
    assert last_name.value == "Hoffmann"
    assert customer_id.value == ""


def test_offscreen_zip_is_revealed_by_scrolling_at_visible_street_label(
    tmp_path: Path, monkeypatch
) -> None:
    state = {"zip_visible": False}

    class Rect:
        def __init__(self, left, top, right, bottom):
            self.left, self.top, self.right, self.bottom = left, top, right, bottom

        def mid_point(self):
            return SimpleNamespace(
                x=(self.left + self.right) // 2,
                y=(self.top + self.bottom) // 2,
            )

    class Label:
        element_info = SimpleNamespace(control_type="Text")

        def __init__(self, text, rect):
            self.text = text
            self.rect = rect

        def window_text(self):
            return self.text

        def rectangle(self):
            return self.rect

        def is_visible(self):
            return self.text == "Street" or state["zip_visible"]

        def is_enabled(self):
            return True

        def scroll_into_view(self):
            raise RuntimeError("SWT does not expose ScrollItem")

    street = Label("Street", Rect(100, 200, 160, 220))
    zip_label = Label("ZIP - City", Rect(100, 500, 160, 520))
    scope = SimpleNamespace(descendants=lambda **kwargs: [street, zip_label])

    class Mouse:
        calls = []

        @classmethod
        def scroll(cls, *, coords, wheel_dist):
            cls.calls.append((coords, wheel_dist))
            state["zip_visible"] = True

    session = SemanticUiaSession(
        tmp_path / "Fakturama.exe", UiaProfile({"window_title_re": ".*"})
    )
    monkeypatch.setattr(
        uia_module,
        "_load_pywinauto",
        lambda: (None, None, Mouse),
    )

    found = session.scroll_until_label(
        ["ZIP"],
        anchor_names=["Street"],
        direction="down",
        scope=scope,
        exact=False,
    )

    assert found is zip_label
    assert Mouse.calls == [((130, 210), -5)]


def test_zip_alias_partially_matches_combined_zip_city_row(tmp_path: Path) -> None:
    class Rect:
        def __init__(self, left, top, right, bottom):
            self.left, self.top, self.right, self.bottom = left, top, right, bottom

    class Element:
        def __init__(self, text, control_type, rect):
            self.text = text
            self.element_info = SimpleNamespace(control_type=control_type)
            self.rect = rect
            self.value = ""

        def window_text(self):
            return self.text

        def rectangle(self):
            return self.rect

        def is_visible(self):
            return True

        def is_enabled(self):
            return True

        def set_edit_text(self, value):
            self.value = value

        def get_value(self):
            return self.value

    label = Element("ZIP - City", "Text", Rect(0, 20, 100, 40))
    zip_field = Element("", "Edit", Rect(110, 20, 180, 40))
    city_field = Element("", "Edit", Rect(190, 20, 400, 40))
    telefax_field = Element("", "Edit", Rect(700, 20, 900, 40))
    elements = [label, telefax_field, city_field, zip_field]
    scope = SimpleNamespace(descendants=lambda **kwargs: elements)
    session = SemanticUiaSession(
        tmp_path / "Fakturama.exe", UiaProfile({"window_title_re": ".*"})
    )

    session.set_labeled_row_values(
        ["ZIP"], ["10115", "Berlin"], scope=scope, exact=False
    )

    assert zip_field.value == "10115"
    assert city_field.value == "Berlin"
    assert telefax_field.value == ""


def test_numeric_and_date_readback_accepts_ui_formatting(
    tmp_path: Path, monkeypatch
) -> None:
    class Field:
        element_info = SimpleNamespace(control_type="Edit")

        def __init__(self):
            self.value = ""

        def set_edit_text(self, value):
            self.value = "14.07.2026" if "2026-07-14" in value else "19,00 %"

        def get_value(self):
            return self.value

    field = Field()
    session = SemanticUiaSession(
        tmp_path / "Fakturama.exe", UiaProfile({"window_title_re": ".*"})
    )
    monkeypatch.setattr(session, "input_for_label", lambda *args, **kwargs: field)

    session.set_labeled_decimal(["Value"], Decimal("19"))
    session.set_labeled_date(
        ["Date"], date(2026, 7, 14), date_format="%Y-%m-%d"
    )


def test_swt_date_is_entered_month_day_year_and_committed(
    tmp_path: Path, monkeypatch
) -> None:
    class Field:
        element_info = SimpleNamespace(control_type="Edit")

        def __init__(self):
            self.value = "Aug 16, 2026"
            self.keys = []
            self.click_coords = None

        def get_value(self):
            return self.value

        def set_edit_text(self, value):
            raise AssertionError("segmented SWT date must use keyboard entry")

        def rectangle(self):
            return SimpleNamespace(left=100, top=50, right=230, bottom=70)

        def click_input(self, *, coords):
            self.click_coords = coords

        def type_keys(self, value, **kwargs):
            self.keys.append(value)
            if value == "{TAB}":
                self.value = "Jul 14, 2026"

    field = Field()
    session = SemanticUiaSession(
        tmp_path / "Fakturama.exe",
        UiaProfile(
            {
                "window_title_re": ".*",
                "date_segment_order": ["month", "day", "year"],
            }
        ),
    )
    monkeypatch.setattr(session, "input_for_label", lambda *args, **kwargs: field)

    session.set_labeled_date(
        ["Date"], date(2026, 7, 14), date_format="%Y-%m-%d"
    )

    assert field.value == "Jul 14, 2026"
    assert field.click_coords == (13, 10)
    assert field.keys == ["7", "{RIGHT}", "14", "{RIGHT}", "2026", "{TAB}"]


def test_follow_up_invoice_action_is_scoped_to_saved_order_group(tmp_path: Path) -> None:
    class Rect:
        def __init__(self, x, y):
            self.point = SimpleNamespace(x=x, y=y)

        def mid_point(self):
            return self.point

    class Element:
        def __init__(self, text, control_type, x, y):
            self.text = text
            self.element_info = SimpleNamespace(control_type=control_type)
            self.rect = Rect(x, y)
            self.invoked = 0

        def window_text(self):
            return self.text

        def rectangle(self):
            return self.rect

        def is_visible(self):
            return True

        def is_enabled(self):
            return True

        def invoke(self):
            self.invoked += 1

    group = Element("Create a follow-up document", "Group", 500, 300)
    toolbar_invoice = Element("Invoice", "Button", 50, 20)
    linked_invoice = Element("Invoice", "Button", 510, 310)
    scope = SimpleNamespace(
        descendants=lambda: [group, toolbar_invoice, linked_invoice]
    )
    session = SemanticUiaSession(
        tmp_path / "Fakturama.exe", UiaProfile({"window_title_re": ".*"})
    )

    session.invoke_in_group(
        ["Create a follow-up document"], ["Invoice"], scope=scope
    )

    assert linked_invoice.invoked == 1
    assert toolbar_invoice.invoked == 0


def test_editor_readiness_requires_tab_not_toolbar_text(tmp_path: Path) -> None:
    class Element:
        def __init__(self, text, control_type):
            self.text = text
            self.element_info = SimpleNamespace(control_type=control_type)

        def window_text(self):
            return self.text

        def is_visible(self):
            return True

        def is_enabled(self):
            return True

    toolbar = Element("Create: New Order", "Button")
    editor = Element("New Order", "TabItem")
    scope = SimpleNamespace(descendants=lambda: [toolbar, editor])
    session = SemanticUiaSession(
        tmp_path / "Fakturama.exe", UiaProfile({"window_title_re": ".*"})
    )

    found = session.wait_for_control(
        ["New Order"], control_types=("TabItem",), scope=scope, timeout=0.1
    )

    assert found is editor


def test_dialog_close_uses_fresh_tree_not_stale_wrapper(
    tmp_path: Path, monkeypatch
) -> None:
    class StaleDialog:
        def window_text(self):
            return "Select the address"

        def is_visible(self):
            raise AssertionError("stale wrapper visibility must not be polled")

    session = SemanticUiaSession(
        tmp_path / "Fakturama.exe", UiaProfile({"window_title_re": ".*"})
    )
    rediscovery = Mock(side_effect=[{}, {}])
    monkeypatch.setattr(session, "_visible_named_windows", rediscovery)

    session.wait_until_closed(StaleDialog(), timeout=0.5)

    assert rediscovery.call_args_list == [
        call(["Select the address"]),
        call(["Select the address"]),
    ]


def test_stale_swt_editor_runtime_id_is_reacquired_by_unique_name(
    tmp_path: Path,
) -> None:
    class Tab:
        element_info = SimpleNamespace(
            control_type="TabItem", runtime_id=(42, 999, 4, -17)
        )

        def __init__(self):
            self.selected = False

        def window_text(self):
            return "*New Order"

        def select(self):
            self.selected = True

    tab = Tab()
    root = SimpleNamespace(descendants=lambda **kwargs: [tab])
    gateway = PywinautoFakturamaGateway(
        tmp_path / "Fakturama.exe", Path("config/fakturama-2.2.0-en.json")
    )
    gateway.session.main = root

    current_id = gateway._activate_editor_tab(
        "uia-42-111-4--17", fallback_names=["New Order"]
    )

    assert tab.selected is True
    assert current_id == "uia-42-999-4--17"


def test_payment_code_combo_selection_with_credit_transfer(tmp_path: Path, monkeypatch) -> None:
    class Combo:
        element_info = SimpleNamespace(control_type="ComboBox")

        def __init__(self):
            self.value = "Cash"
            self.selected = ""

        def window_text(self):
            return self.value

        def is_visible(self):
            return True

        def is_enabled(self):
            return True

        def item_texts(self):
            return ["Cash", "Credit transfer", "Credit card", "SEPA direct debit"]

        def select(self, value):
            self.selected = value
            self.value = value

        def get_value(self):
            return self.value

    class Edit:
        element_info = SimpleNamespace(control_type="Edit")

        def __init__(self, value=""):
            self.value = value

        def window_text(self):
            return self.value

        def is_visible(self):
            return True

        def is_enabled(self):
            return True

        def set_edit_text(self, value):
            self.value = value

        def get_value(self):
            return self.value

    class Label:
        element_info = SimpleNamespace(control_type="Text")

        def __init__(self, text):
            self.text = text

        def window_text(self):
            return self.text

        def is_visible(self):
            return True

        def is_enabled(self):
            return False  # Static labels often report is_enabled=False in UIA

    combo = Combo()
    session = SemanticUiaSession(
        tmp_path / "Fakturama.exe", UiaProfile.load(Path("config/fakturama-2.2.0-en.json"))
    )

    scope = SimpleNamespace(descendants=lambda: [combo])
    session.select_semantic_option(
        ["Payment code", "Code"],
        ["Credit transfer"],
        distinguishing_options=["Credit card", "SEPA direct debit"],
        scope=scope,
    )

    assert combo.value == "Credit transfer"


def test_save_once_falls_back_to_ctrl_s_when_button_missing(tmp_path: Path, monkeypatch) -> None:
    gateway = PywinautoFakturamaGateway(
        tmp_path / "Fakturama.exe", Path("config/fakturama-2.2.0-en.json")
    )
    sent_keys = []

    class MockWindow:
        def type_keys(self, keys):
            sent_keys.append(keys)

    win = MockWindow()
    monkeypatch.setattr(gateway.session, "invoke", Mock(side_effect=uia_module.UnsupportedAutomation("no button")))
    monkeypatch.setattr(gateway.session, "active_window", lambda: win)
    monkeypatch.setattr(gateway.session, "root", lambda: win)

    gateway._save_once()

    assert sent_keys == ["^s"]


def test_debtor_editor_tab_reacquisition_with_company_name(tmp_path: Path) -> None:
    class Tab:
        element_info = SimpleNamespace(
            control_type="TabItem", runtime_id=(42, 6424740, 4, -17)
        )

        def __init__(self):
            self.selected = False

        def window_text(self):
            return "*Lindgren Industrial Supplies B.V."

        def select(self):
            self.selected = True

    tab = Tab()
    root = SimpleNamespace(descendants=lambda **kwargs: [tab])
    gateway = PywinautoFakturamaGateway(
        tmp_path / "Fakturama.exe", Path("config/fakturama-2.2.0-en.json")
    )
    gateway.session.main = root
    gateway._pending_debtor = SimpleNamespace(
        company="Lindgren Industrial Supplies B.V.",
        first_name="Sanne",
        last_name="Verhoeven",
    )

    current_id = gateway._activate_editor_tab(
        "stale-id-999", fallback_names=gateway._debtor_fallback_names()
    )

    assert tab.selected is True
    assert current_id == "uia-42-6424740-4--17"



