from src.utils.utils import label_count_template


class LabelCounter:
    def __init__(self):
        self.label_count = label_count_template

        self.label_threshold = 3

    def update_label_count(self, label):
        if label in self.label_count:
            if self.label_count[label] is None:
                self.label_count[label] = 1
            else:
                self.label_count[label] += 1
        else:
            print(f"Label '{label}' not found in the dictionary")

    def check_label_count(self, label):
        count = self.label_count.get(label, None)
        if count is None:
            return "Label not found or count is None"
        return count >= self.label_threshold

    def reset_all_counts(self):
        for key in self.label_count.keys():
            self.label_count[key] = None
