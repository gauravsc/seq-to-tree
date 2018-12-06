


def f1_score(true_labels, pred_labels):
	true_labels = set(true_labels)
	pred_labels = set(pred_labels)

	# compute precision, recall and f1 scores for one instance
	precision = len(true_labels.intersection(pred_labels))/float(len(pred_labels))
	recall = len(true_labels.intersection(pred_labels))/float(len(true_labels))
	f1_score = 2*precision*recall/(precision+recall)

	return f1_score


def precision(true_labels, pred_labels):

	true_labels = set(true_labels)
	pred_labels = set(pred_labels)
	precision = len(true_labels.intersection(pred_labels))/float(len(pred_labels))

	return precision


def recall(true_labels, pred_labels):
	
	true_labels = set(true_labels)
	pred_labels = set(pred_labels)
	recall = len(true_labels.intersection(pred_labels))/float(len(true_labels))

	return recall




