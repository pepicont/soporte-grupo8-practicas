from kivy.app import App
from kivy.uix.tabbedpanel import TabbedPanel


class TabbedPannel(TabbedPanel):
    pass


class TabbedPannelApp(App):
    def build(self):
        return TabbedPannel()


if __name__ == '__main__':
    TabbedPannelApp().run()