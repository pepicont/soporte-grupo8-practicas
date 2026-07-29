from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.properties import StringProperty


class CalculatorLogic:
    def __init__(self):
        self.last_result = None
        self.reset()

    def reset(self):
        self.current     = "0"
        self.expression  = ""
        self.operator    = None
        self.prev_value  = None
        self.just_evaled = False

    def input_digit(self, digit):
        if self.just_evaled:
            self.current     = digit
            self.expression  = ""
            self.just_evaled = False
        elif self.current == "0":
            self.current = digit
        elif len(self.current) < 15:
            self.current += digit

    def input_dot(self):
        if self.just_evaled:
            self.current     = "0."
            self.just_evaled = False
        elif "." not in self.current:
            self.current += "."

    def input_operator(self, op):
        self.just_evaled = False
        current_f = float(self.current)
        if self.prev_value is not None and self.operator:
            result          = self._compute(self.prev_value, current_f, self.operator)
            self.current    = self._fmt(result)
            self.prev_value = result
        else:
            self.prev_value = current_f
        self.operator   = op
        self.expression = f"{self._fmt(self.prev_value)} {op}"
        self.current    = "0"

    def evaluate(self):
        if self.prev_value is None or self.operator is None:
            return None
        b      = float(self.current)
        expr   = f"{self._fmt(self.prev_value)} {self.operator} {self._fmt(b)}"
        result = self._compute(self.prev_value, b, self.operator)
        self.last_result = result          # <-- guardamos para ANS
        self.expression  = expr + " ="
        self.current     = self._fmt(result)
        self.prev_value  = None
        self.operator    = None
        self.just_evaled = True
        return result

    def recall_ans(self):
        """Carga el último resultado como valor actual."""
        if self.last_result is None:
            return
        ans_str = self._fmt(self.last_result)
        if self.just_evaled:
            self.current     = ans_str
            self.expression  = f"ANS ({ans_str})"
            self.just_evaled = False
        else:
            # si ya hay algo escrito, reemplazamos el current
            self.current    = ans_str
            self.expression = f"ANS ({ans_str})"

    def backspace(self):
        if self.just_evaled:
            return
        self.current = self.current[:-1] if len(self.current) > 1 else "0"

    def toggle_sign(self):
        if self.current.startswith("-"):
            self.current = self.current[1:]
        elif self.current != "0":
            self.current = "-" + self.current

    @staticmethod
    def _compute(a, b, op):
        if op == "+": return a + b
        if op == "-": return a - b
        if op == "x": return a * b
        if op == "/":
            if b == 0:
                raise ZeroDivisionError
            return a / b
        return b

    @staticmethod
    def _fmt(v):
        return str(int(v)) if isinstance(v, float) and v == int(v) else f"{v:.10g}"


class CalculatorRoot(BoxLayout):
    display_expr   = StringProperty("")
    display_result = StringProperty("0")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.logic = CalculatorLogic()

    def press_digit(self, d):
        self.logic.input_digit(d)
        self._refresh()

    def press_dot(self):
        self.logic.input_dot()
        self._refresh()

    def press_op(self, op):
        self.logic.input_operator(op)
        self._refresh()

    def press_equal(self):
        try:
            self.logic.evaluate()
        except ZeroDivisionError:
            self.display_expr   = "Error"
            self.display_result = "/ 0"
            self.logic.reset()
            return
        self._refresh()

    def press_ac(self):
        self.logic.reset()
        self._refresh()

    def press_sign(self):
        self.logic.toggle_sign()
        self._refresh()

    def press_back(self):
        self.logic.backspace()
        self._refresh()

    def press_ans(self):
        self.logic.recall_ans()
        self._refresh()

    def _refresh(self):
        self.display_expr   = self.logic.expression
        self.display_result = self.logic.current


class CalculatorApp(App):
    def build(self):
        from kivy.core.window import Window
        Window.size = (360, 620)
        return CalculatorRoot()