import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE

from utils import get_bounds


def plot_feature_distribution(
        sample_features,
        ax=None,
        t_sne=True,
        hue=None,
        hue_order=None,
        style=None,
        style_order=None,
        x_lim='min_max',
        y_lim='min_max',
        contour=False,
        z_generator=None):
    if ax is None:
        raise AttributeError('you must specify an ax')
    sample_features = np.array(sample_features)
    assert sample_features.ndim == 2
    assert x_lim in ['min_max', 'min_max_extend', 'box'] or len(x_lim) == 2
    assert y_lim in ['min_max', 'min_max_extend', 'box'] or len(y_lim) == 2

    ndim = sample_features.shape[1]
    if ndim != 2 and not t_sne:
        raise AttributeError('must use t_sne to preprocess features whose ndim is not 2')
    if ndim == 1:
        sample_features = np.concatenate([sample_features, np.zeros_like(sample_features)], 1)
        ndim = 2

    if t_sne and ndim != 2:
        sample_features = TSNE(n_components=2, init='pca', random_state=1, n_jobs=4, method='exact')\
            .fit_transform(sample_features)
    lower_bound_x, upper_bound_x = x_lim if len(x_lim) == 2 else \
        get_bounds(sample_features[:, 0], x_lim, extend_factor=0.2)
    lower_bound_y, upper_bound_y = y_lim if len(y_lim) == 2 else \
        get_bounds(sample_features[:, 1], y_lim, extend_factor=0.1)

    if contour:
        plot_contour([lower_bound_x, upper_bound_x], [lower_bound_y, upper_bound_y], z_generator, ax=ax)
    sns.scatterplot(x=sample_features[:, 0], y=sample_features[:, 1],
                    hue=hue,
                    hue_order=hue_order,
                    style=style,
                    style_order=style_order,
                    ax=ax)

    ax.legend(loc="best")
    ax.set_xlim(lower_bound_x, upper_bound_x)
    ax.set_ylim(lower_bound_y, upper_bound_y)


def plot_contour(x_range, y_range, z_generator, resolution=500, alpha=0.2, ax=None):
    assert ax is not None

    lower_bound_x, upper_bound_x = x_range
    lower_bound_y, upper_bound_y = y_range
    xx, yy = np.meshgrid(np.linspace(lower_bound_x, upper_bound_x, resolution),
                         np.linspace(lower_bound_y, upper_bound_y, resolution))
    z = z_generator(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, z, alpha=alpha)


def plot_table(table_val_list, row_labels, cell_loc='center', table_loc='upper center', ax=None, additional_text=None):
    assert ax is not None

    ax.axis('off')
    table_content = []
    for table_val_item in table_val_list:
        if isinstance(table_val_item[0], float):
            table_content.append(['%1.1f' % item for item in table_val_item])
        else:
            table_content.append(['%s' % item for item in table_val_item])
    ax.table(cellText=table_content, rowLabels=row_labels, cellLoc=cell_loc, loc=table_loc)
    if additional_text is not None:
        if not isinstance(additional_text, list):
            additional_text = [additional_text]
        locations = np.linspace(0.5, 0.1, len(additional_text))
        for add_text, loc in zip(additional_text, locations):
            ax.text(0, loc, add_text, wrap=True, clip_on=True)
