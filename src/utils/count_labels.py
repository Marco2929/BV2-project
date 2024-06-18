from src.utils.utils import label_count_template


class LabelCounter:
    def __init__(self):
        """
        Initializes the LabelCounter with a label count dictionary and a threshold for detection.
        """
        self.label_count = label_count_template

        self.label_threshold = 3

    def update_label_count(self, label: str):
        """
        Updates the count of the given label. If the label is found in the dictionary,
        increments its count. If the label is not found, prints an error message.

        Args:
            label (str): The label to be updated.
        """
        if label in self.label_count:
            if self.label_count[label] is None:
                self.label_count[label] = 1
            else:
                self.label_count[label] += 1
        else:
            print(f"Label '{label}' not found in the dictionary")

    def check_label_count(self, label: str) -> bool:
        """
        Checks if the count of the given label has reached the threshold.

        Args:
            label (str): The label to be checked.

        Returns:
            bool: True if the count is greater than or equal to the threshold, False otherwise.
            str: Message if the label is not found or count is None.
        """
        count = self.label_count.get(label, None)
        if count is None:
            raise "Label not found or count is None"
        return count >= self.label_threshold

    def reset_all_counts(self):
        """
            Reset all counts to use counter again
        """
        for key in self.label_count.keys():
            self.label_count[key] = None
