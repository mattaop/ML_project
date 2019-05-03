from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

from feature_extraction.data_processing import scale_input, grey_scale, remove_dimension
from feature_extraction.filters import add_edge_detection_filter, add_median_filter, \
    add_bilateral_filter, add_gaussian_filter, add_otsu_filter
from load_data import load_data_chars


def svm(x, y):
    """Fitting SVM model"""
    print('=== Non-linear SVM classification ===')
    random_state = 100
    # Data processing
    x = scale_input(x)
    x = grey_scale(x)
    # x = add_edge_detection_filter(x)
    x = add_otsu_filter(x)
    x = remove_dimension(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,
                                                        random_state=random_state,
                                                        stratify=y)

    # Fit model
    clf = SVC(gamma='scale', kernel='rbf')
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print('Test accuracy: ', accuracy_score(y_test, y_pred))
    return clf


if __name__ == "__main__":
    img, target = load_data_chars()
    svm(img, target)
