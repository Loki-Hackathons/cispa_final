"""Abstract dashboard data provider."""

from abc import ABC, abstractmethod

from dashboard.models import DashboardStatus


class StatusProvider(ABC):
    @abstractmethod
    def get_status(self) -> DashboardStatus:
        pass
