class ThresholdLabelEncoder:
    def __init__(self, classes: list[str], index: int = 1):
        self.mapping = {cls: i + index for i, cls in enumerate(classes)}
        self.index = index

    def __post_init__(self):
        if len(self.mapping) <= 1:
            raise ValueError("Mapping must contain at least two classes.")

    def _get_transposed_mapping(self) -> dict[int, str]:
        return {v: k for k, v in self.mapping.items()}

    def get_class_by_thresholds(self, value: float, thresholds: list[float]) -> str:
        if len(thresholds) != len(self.mapping) - 1:
            raise ValueError("Thresholds list cannot be empty.")
        if thresholds != sorted(thresholds):
            raise ValueError("Thresholds must be sorted in ascending order.")
        transposed_mapping = self._get_transposed_mapping()
        for i, threshold in enumerate(thresholds):
            if value < threshold:
                return transposed_mapping[i + self.index]
        return transposed_mapping[len(self.mapping) - 1 + self.index]
