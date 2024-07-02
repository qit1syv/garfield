from nerfstudio.viewer.viewer_elements import ViewerParameter, ViewerText, ViewerElement

class ViserMarkdown(ViewerElement):
    """A text field in the viewer

    Args:
        name: The name of the text field
        default_value: The default value of the text field
        disabled: If the text field is disabled
        visible: If the text field is visible
        cb_hook: Callback to call on update
        hint: The hint text
    """

    def __init__(self, name: str, cb_hook=lambda x: None, disabled: bool = False, visible: bool = True):
        super().__init__(name, disabled=disabled, visible=visible, cb_hook=cb_hook)

    def _create_gui_handle(self, viser_server) -> None:
        self.gui_handle = viser_server.gui.add_markdown("Your VLM Output will be shown here")

    def install(self, viser_server) -> None:
        self._create_gui_handle(viser_server)

        assert self.gui_handle is not None
