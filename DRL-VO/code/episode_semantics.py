class EnvironmentEpisodeFault(RuntimeError):
 def __init__(self,reason,diagnostics=None):
  self.reason=reason;self.diagnostics=diagnostics;super().__init__(reason)
class RobotEpisodeTerminal(Exception):
 def __init__(self,reason):self.reason=reason;super().__init__(reason)
