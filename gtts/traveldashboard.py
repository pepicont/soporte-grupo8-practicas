from kivy.app import App
from kivy.clock import mainthread
from kivy.factory import Factory
from kivy.uix.boxlayout import BoxLayout

from tts_client import speak

PHRASES = [
    ('¿Dónde está el baño?', 'Where is the bathroom?'),
    ('¿Cuánto cuesta esto?', 'How much does this cost?'),
    ('Necesito ayuda, por favor', 'I need help, please'),
    ('¿Dónde queda el hotel?', 'Where is the hotel?'),
    ('Quisiera pedir esto', 'I would like to order this'),
    ('¿Habla español?', 'Do you speak Spanish?'),
    ('Estoy perdido', 'I am lost'),
    ('Gracias, muy amable', 'Thank you, very kind'),
]


class TravelDashboard(BoxLayout):
    def say_phrase(self, english_text):
        for child in self.ids.buttons_grid.children:
            child.disabled = True

        @mainthread
        def show_error(message):
            popup = Factory.Popup(
                title='Error',
                content=Factory.Label(text=message),
                size_hint=(0.7, 0.4),
            )
            popup.open()

        @mainthread
        def re_enable():
            for child in self.ids.buttons_grid.children:
                child.disabled = False

        speak(english_text, on_error=show_error, on_end=re_enable)


class TravelDashboardApp(App):
    def build(self):
        dashboard = TravelDashboard()
        for spanish_text, english_text in PHRASES:
            button = Factory.Button(text=spanish_text)
            button.bind(
                on_press=lambda btn, en=english_text: dashboard.say_phrase(en)
            )
            dashboard.ids.buttons_grid.add_widget(button)
        return dashboard


if __name__ == '__main__':
    TravelDashboardApp().run()
