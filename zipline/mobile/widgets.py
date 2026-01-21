"""Mobile widget data endpoints"""


class MobileWidgets:
    """Widget data for iOS and Android."""
    
    def get_widget_data(self, widget_type: str) -> dict:
        """Get widget data."""
        return {'type': widget_type, 'data': {}}
