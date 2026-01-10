"""Voice Trading Interface"""


class VoiceTradingInterface:
    """Voice command support for trading."""
    
    def __init__(self):
        self.wake_word = "hey zipline"
    
    def process_voice_command(self, audio_data: bytes) -> str:
        """Process voice command."""
        return "command text"
