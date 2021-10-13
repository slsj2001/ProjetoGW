import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import pandas as pd
import patsy
import sklearn.linear_model as linear
import random
from collections import defaultdict
from sklearn import preprocessing
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor as RFR


#  Functions for Single Variable EDA


def helper_describe_numerical_variable(data, key):
    # Describe the variable
    print('min = {stat:.2f}'.format(stat=np.min(data[key])))
    print('Q1 = {stat:.2f}'.format(
        stat=stats.mstats.mquantiles(data[key], [0.25])[0]))
    print('mean = {stat:.2f}'.format(stat=np.mean(data[key])))
    print('median (Q2) = {stat:.2f}'.format(stat=np.median(data[key])))
    print('Q3 = {stat:.2f}'.format(
        stat=stats.mstats.mquantiles(data[key], [0.75])[0]))
    print('max = {stat:.2f}'.format(stat=np.max(data[key])))
    print('range = {stat:.2f}'.format(
        stat=np.max(data[key]) - np.min(data[key])))
    print('IQR = {stat:.2f}'.format(
        stat=stats.mstats.mquantiles(data[key], [0.75])[0] -
             stats.mstats.mquantiles(data[key], [0.25])[0]))
    print(
        'variance (std) = {var:.2f} ({std:.2f})'.format(var=np.var(data[key]),
                                                        std=np.std(data[key])))
    print('COV = {stat:.2f}%'.format(
        stat=(np.std(data[key]) / np.mean(data[key])) * 100))
    # Plot the histogram
    figure, axes = plt.subplots()
    axes.hist(data[key], density=True)
    axes.axvline(np.mean(data[key]), color='DarkRed')
    axes.axvline(np.median(data[key]), color='DarkOrange')
    axes.set_xlabel(key)
    axes.set_title('{} Distribution'.format(key))
    plt.show()
    plt.close()


#  Functions for Pairwise EDA


def restyle_boxplot(patch):
    ## change color and linewidth of the whiskers
    for whisker in patch['whiskers']:
        whisker.set(color='#000000', linewidth=1)

    ## change color and linewidth of the caps
    for cap in patch['caps']:
        cap.set(color='#000000', linewidth=1)

    ## change color and linewidth of the medians
    for median in patch['medians']:
        median.set(color='#000000', linewidth=2)

    ## change the style of fliers and their fill
    for flier in patch['fliers']:
        flier.set(marker='o', color='#000000', alpha=0.2)

    for box in patch["boxes"]:
        box.set(facecolor='#FFFFFF', alpha=0.5)


def multiboxplot(data, numeric, categorical, skip_data_points=True):
    figure = plt.figure(figsize=(10, 6))

    axes = figure.add_subplot(1, 1, 1)

    grouped = data.groupby(categorical)
    labels = pd.unique(data[categorical].values)
    labels.sort()
    grouped_data = [grouped[numeric].get_group(k) for k in labels]
    patch = axes.boxplot(grouped_data, labels=labels, patch_artist=True,
                         zorder=1)
    restyle_boxplot(patch)

    if not skip_data_points:
        for i, k in enumerate(labels):
            subdata = grouped[numeric].get_group(k)
            x = np.random.normal(i + 1, 0.01, size=len(subdata))
            axes.plot(x, subdata, 'o', alpha=0.4, color="DimGray", zorder=2)

    axes.set_xlabel(categorical)
    axes.set_ylabel(numeric)
    axes.set_title("Distribution of {0} by {1}".format(numeric, categorical))

    plt.show()
    plt.close()


def lowess_scatter(data, x, y, jitter=0.0, skip_lowess=True):
    if skip_lowess:
        fit = np.polyfit(data[x], data[y], 1)
        line_x = np.linspace(data[x].min(), data[x].max(), 10)
        line = np.poly1d(fit)
        line_y = list(map(line, line_x))
    else:
        lowess = sm.nonparametric.lowess(data[y], data[x], frac=.3)
        line_x = list(zip(*lowess))[0]
        line_y = list(zip(*lowess))[1]

    figure = plt.figure(figsize=(10, 6))

    axes = figure.add_subplot(1, 1, 1)

    xs = data[x]
    if jitter > 0.0:
        xs = data[x] + stats.norm.rvs(0, 0.5, data[x].size)

    axes.scatter(xs, data[y], marker="o", color="DimGray", alpha=0.5)
    axes.plot(line_x, line_y, color="DarkRed")

    title = "Plot of {0} v. {1}".format(x, y)
    if not skip_lowess:
        title += " with LOWESS"
    axes.set_title(title)
    axes.set_xlabel(x)
    axes.set_ylabel(y)

    plt.show()
    plt.close()


def correlation(data, x, y):
    print("Correlation coefficients:")
    print("r   =", stats.pearsonr(data[x], data[y])[0])
    print("rho =", stats.spearmanr(data[x], data[y])[0])


from IPython.display import HTML, display_html

ALGORITHMS = {
    "linear": linear.LinearRegression,
    "ridge": linear.Ridge,
    "lasso": linear.Lasso
}


def summarize(formula, X, y, model, style='linear'):
    result = {}
    result["formula"] = formula
    result["n"] = len(y)
    result["model"] = model
    # I think this is a bug in Scikit Learn
    # because lasso should work with multiple targets.
    if style == "lasso":
        result["coefficients"] = model.coef_
    else:
        result["coefficients"] = model.coef_[0]
    result["r_squared"] = model.score(X, y)
    y_hat = model.predict(X)
    result["residuals"] = y - y_hat
    result["y_hat"] = y_hat
    result["y"] = y
    sum_squared_error = sum([e ** 2 for e in result["residuals"]])[0]

    n = len(result["residuals"])
    k = len(result["coefficients"])

    result["sigma"] = np.sqrt(sum_squared_error / (n - k))
    return result


def linear_regression(formula, data=None, style="linear", params={}):
    if data is None:
        raise ValueError(
            "The parameter 'data' must be assigned a non-nil reference to a Pandas DataFrame")

    params["fit_intercept"] = False

    y, X = patsy.dmatrices(formula, data, return_type="matrix")
    algorithm = ALGORITHMS[style]
    algo = algorithm(**params)
    model = algo.fit(X, y)

    result = summarize(formula, X, y, model, style)

    # result = {}
    # result["formula"] = formula
    # result["n"] = data.shape[ 0]

    # result["model"] = model

    # if style == "lasso":
    #     result["coefficients"] = model.coef_
    # else:
    #     result["coefficients"] =  model.coef_[0]

    # result["r_squared"] = model.score( X, y)

    # y_hat = model.predict( X)
    # result["residuals"] = y - y_hat
    # result["y_hat"] = y_hat
    # result["y"]  = y
    # sum_squared_error = sum([e**2 for e in result[ "residuals"]])[0]

    # n = len(result["residuals"])
    # k = len(result["coefficients"])

    # result["sigma"] = np.sqrt( sum_squared_error / (n - k))

    return result


def logistic(z):
    return 1.0 / (1.0 + np.exp(-z))


def logistic_regression(formula, data=None):
    if data is None:
        raise ValueError(
            "The parameter 'data' must be assigned a non-nil reference to a Pandas DataFrame")

    result = {}
    result["formula"] = formula
    result["n"] = data.shape[0]

    y, X = patsy.dmatrices(formula, data, return_type="matrix")
    y = np.ravel(
        y)  # not sure why this is needed for LogisticRegression but not LinearRegression

    model = linear.LogisticRegression(fit_intercept=False).fit(X, y)
    result["model"] = model

    result["coefficients"] = model.coef_[0]

    y_hat = model.predict(X)
    result["residuals"] = y - y_hat
    result["y_hat"] = y_hat
    result["y"] = y

    # efron's pseudo R^2
    y_bar = np.mean(y)
    pr = model.predict_proba(X).transpose()[1]
    result["probabilities"] = pr
    efrons_numerator = np.sum((y - pr) ** 2)
    efrons_denominator = np.sum((y - y_bar) ** 2)
    result["r_squared"] = 1 - (efrons_numerator / efrons_denominator)

    # error rate
    result["sigma"] = np.sum(np.abs(result["residuals"])) / result["n"] * 100

    n = len(result["residuals"])
    k = len(result["coefficients"])

    return result


def bootstrap_linear_regression(formula, data=None, samples=100,
                                style="linear", params={}):
    if data is None:
        raise ValueError(
            "The parameter 'data' must be assigned a non-nil reference to a Pandas DataFrame")

    bootstrap_results = {}
    bootstrap_results["formula"] = formula

    variables = [x.strip() for x in formula.split("~")[1].split("+")]
    variables = ["intercept"] + variables
    bootstrap_results["variables"] = variables

    coeffs = []
    sigmas = []
    rs = []

    n = data.shape[0]
    bootstrap_results["n"] = n

    for i in range(samples):
        sampling_indices = [i for i in
                            [np.random.randint(0, n - 1) for _ in range(0, n)]]
        sampling = data.loc[sampling_indices]

        results = linear_regression(formula, data=sampling, style=style,
                                    params=params)
        coeffs.append(results["coefficients"])
        sigmas.append(results["sigma"])
        rs.append(results["r_squared"])

    coeffs = pd.DataFrame(coeffs, columns=variables)
    sigmas = pd.Series(sigmas, name="sigma")
    rs = pd.Series(rs, name="r_squared")

    bootstrap_results["resampled_coefficients"] = coeffs
    bootstrap_results["resampled_sigma"] = sigmas
    bootstrap_results["resampled_r^2"] = rs

    result = linear_regression(formula, data=data)

    bootstrap_results["residuals"] = result["residuals"]
    bootstrap_results["coefficients"] = result["coefficients"]
    bootstrap_results["sigma"] = result["sigma"]
    bootstrap_results["r_squared"] = result["r_squared"]
    bootstrap_results["model"] = result["model"]
    bootstrap_results["y"] = result["y"]
    bootstrap_results["y_hat"] = result["y_hat"]
    return bootstrap_results


def bootstrap_logistic_regression(formula, data=None, samples=100):
    if data is None:
        raise ValueError(
            "The parameter 'data' must be assigned a non-nil reference to a Pandas DataFrame")

    bootstrap_results = {}
    bootstrap_results["formula"] = formula

    variables = [x.strip() for x in formula.split("~")[1].split("+")]
    variables = ["intercept"] + variables
    bootstrap_results["variables"] = variables

    coeffs = []
    sigmas = []
    rs = []

    n = data.shape[0]
    bootstrap_results["n"] = n

    for i in range(samples):
        sampling_indices = [i for i in
                            [np.random.randint(0, n - 1) for _ in range(0, n)]]
        sampling = data.loc[sampling_indices]

        results = logistic_regression(formula, data=sampling)
        coeffs.append(results["coefficients"])
        sigmas.append(results["sigma"])
        rs.append(results["r_squared"])

    coeffs = pd.DataFrame(coeffs, columns=variables)
    sigmas = pd.Series(sigmas, name="sigma")
    rs = pd.Series(rs, name="r_squared")

    bootstrap_results["resampled_coefficients"] = coeffs
    bootstrap_results["resampled_sigma"] = sigmas
    bootstrap_results["resampled_r^2"] = rs

    result = logistic_regression(formula, data=data)

    bootstrap_results["residuals"] = result["residuals"]
    bootstrap_results["coefficients"] = result["coefficients"]
    bootstrap_results["sigma"] = result["sigma"]
    bootstrap_results["r_squared"] = result["r_squared"]
    bootstrap_results["model"] = result["model"]
    return bootstrap_results


def fmt(n, sd=2):
    return (r"{0:." + str(sd) + "f}").format(n)


def results_table(fit, sd=2, bootstrap=False, is_logistic=False):
    result = {}
    result["model"] = [fit["formula"]]

    variables = [""] + fit["formula"].split("~")[1].split("+")
    coefficients = []

    if bootstrap:
        bounds = fit["resampled_coefficients"].quantile([0.025, 0.975])
        bounds = bounds.transpose()
        bounds = bounds.values.tolist()
        for i, b in enumerate(zip(variables, fit["coefficients"], bounds)):
            coefficient = [b[0], r"$\beta_{0}$".format(i), fmt(b[1], sd),
                           fmt(b[2][0], sd), fmt(b[2][1], sd)]
            if is_logistic:
                if i == 0:
                    coefficient.append(fmt(logistic(b[1]), sd))
                else:
                    coefficient.append(fmt(b[1] / 4, sd))
            coefficients.append(coefficient)
    else:
        for i, b in enumerate(zip(variables, fit["coefficients"])):
            coefficients.append(
                [b[0], r"$\beta_{0}$".format(i), fmt(b[1], sd)])
    result["coefficients"] = coefficients

    error = r"$\sigma$"
    r_label = r"$R^2$"
    if is_logistic:
        error = "Error ($\%$)"
        r_label = r"Efron's $R^2$"
    if bootstrap:
        sigma_bounds = stats.mstats.mquantiles(fit["resampled_sigma"],
                                               [0.025, 0.975])
        r_bounds = stats.mstats.mquantiles(fit["resampled_r^2"],
                                           [0.025, 0.975])
        metrics = [
            [error, fmt(fit["sigma"], sd), fmt(sigma_bounds[0], sd),
             fmt(sigma_bounds[1], sd)],
            [r_label, fmt(fit["r_squared"], sd), fmt(r_bounds[0], sd),
             fmt(r_bounds[1], sd)]]
    else:
        metrics = [
            [error, fmt(fit["sigma"], sd)],
            [r_label, fmt(fit["r_squared"], sd)]]

    result["metrics"] = metrics

    return result


class ResultsView(object):
    def __init__(self, content, bootstrap=False, is_logistic=False):
        self.content = content
        self.bootstrap = bootstrap
        self.is_logistic = is_logistic

    def _repr_html_(self):
        span = "2"
        if self.bootstrap and not self.is_logistic:
            span = "3"
        if self.bootstrap and self.is_logistic:
            span = "5"
        result = r"<table><tr><th colspan=" + span + r">Linear Regression Results</th></tr>"
        if self.is_logistic:
            result = r"<table><tr><th colspan=" + span + r">Logistic Regression Results</th></tr>"
        result += r"<th colspan=" + span + r">Coefficients</th></tr>"
        coefficients = self.content["coefficients"]
        template = r""
        headers = r""
        if self.is_logistic:
            if self.bootstrap:
                header = r"<tr><th>$\theta$</th><th></th><th>95% BCI</th><th>P(y=1)</th></tr>"
                template = r"<tr><td>{0} ({1})</td><td>{2}</td><td>({3}, {4})</td><td>{5}</td></tr>"
            else:
                header = r"<tr><th>$\theta$</th><th></th></tr>"
                template = r"<tr><td>{0} ({1})</td><td>{2}</td></tr>"
        else:
            if self.bootstrap:
                header = r"<tr><th>$\theta$</th><th></th><th>95% BCI</th></tr>"
                template = r"<tr><td>{0} ({1})</td><td>{2}</td><td>({3}, {4})</td></tr>"
            else:
                header = r"<tr><th>$\theta$</th><th></th></tr>"
                template = r"<tr><td>{0} ({1})</td><td>{2}</td></tr>"
        result += header
        for coefficient in coefficients:
            result += template.format(*coefficient)

        result += r"<tr><th colspan=" + span + ">Metrics</th></tr>"

        metrics = self.content["metrics"]
        template = r"<tr><td>{0}</td><td>{1}</td></tr>"
        if self.bootstrap:
            template = r"<tr><td>{0}</td><td>{1}</td><td>({2}, {3})</td><td></td></tr>"

        for metric in metrics:
            result += template.format(*metric)

        result += r"</table>"
        return result

    def _repr_latex_(self):
        span = 2
        if self.bootstrap and not self.is_logistic:
            span = 3
        if self.bootstrap and self.is_logistic:
            span = 4
        result = r"\begin{table}[!htbp] \begin{tabular}{" + (
                r"l" * span) + r"} \hline \multicolumn{" + str(
            span) + r"}{c}{\textbf{Linear Regression}} \\ \hline \hline "
        if self.is_logistic:
            result = r"\begin{table}[!htbp] \begin{tabular}{" + (
                    r"l" * span) + r"} \hline \multicolumn{" + str(
                span) + r"}{c}{\textbf{Logistic Regression}} \\ \hline \hline "

        result += r"\multicolumn{" + str(
            span) + r"}{l}{\textbf{Coefficients}}        \\ \hline "
        coefficients = self.content["coefficients"]
        template = r""
        headers = r""
        if self.is_logistic:
            if self.bootstrap:
                header = r"$\theta$       &          & 95\% BCI     & P(y=1)\\"
                template = r"{0} ({1})      & {2}   & ({3}, {4})   & {5}  \\"
            else:
                header = r"$\theta$                  &                    \\"
                template = r"{0} ({1})                & {2}               \\"
        else:
            if self.bootstrap:
                header = r"$\theta$       &          & 95\% BCI           \\"
                template = r"{0} ({1})      & {2}   & ({3}, {4})          \\"
            else:
                header = r"$\theta$                  &                    \\"
                template = r"{0} ({1})                & {2}               \\"
        result += header
        for coefficient in coefficients:
            coefficient[0] = coefficient[0].replace('_', '\_')
            result += template.format(*coefficient)

        result += r"\hline \multicolumn{" + str(
            span) + r"}{l}{\textbf{Metrics}}             \\ \hline "

        metrics = self.content["metrics"]
        template = r"{0}                & {1}               \\"
        if self.bootstrap:
            template = r"{0}      & {1}   & ({2}, {3})          \\"

        for metric in metrics:
            result += template.format(*metric)
        result += r"\hline"
        result += r"\end{tabular}\end{table}"
        return result


def print_csv(table):
    print("Linear Regression")
    print("Coefficients")
    for item in table["coefficients"]:
        print(','.join(item))
    print("Metrics")
    for item in table["metrics"]:
        print(','.join(item))


def simple_describe_lr(fit, sd=2):
    table = results_table(fit, sd)
    return ResultsView(table)


def simple_describe_lgr(fit, sd=2):
    table = results_table(fit, sd, False, True)
    return ResultsView(table, False, True)


def describe_bootstrap_lr(fit, sd=2):
    table = results_table(fit, sd, True, False)
    return ResultsView(table, True, False)


def describe_bootstrap_lgr(fit, sd=2):
    table = results_table(fit, sd, True, True)
    return ResultsView(table, True, True)


def strength(pr):
    if 0 <= pr <= 0.33:
        return "weak"
    if 0.33 < pr <= 0.66:
        return "mixed"
    return "strong"


# {"var1": "+", "var2": "-"}
def evaluate_coefficient_predictions(predictions, result):
    coefficients = result["resampled_coefficients"].columns
    for coefficient in coefficients:
        if coefficient == 'intercept':
            continue
        if predictions[coefficient] == '+':
            pr = np.mean(result["resampled_coefficients"][coefficient] > 0)
            print("{0} P(>0)={1:.3f} ({2})".format(coefficient, pr,
                                                   strength(pr)))
        else:
            pr = np.mean(result["resampled_coefficients"][coefficient] < 0)
            print("{0} P(<0)={1:.3f} ({2})".format(coefficient, pr,
                                                   strength(pr)))


def adjusted_r_squared(result):
    adjustment = (result["n"] - 1) / (
            result["n"] - len(result["coefficients"]) - 1 - 1)
    return 1 - (1 - result["r_squared"]) * adjustment


def plot_residuals(result, variables, data):
    figure = plt.figure(figsize=(20, 6))

    plots = len(variables)
    rows = (plots // 3) + 1

    residuals = np.array([r[0] for r in result["residuals"]])
    limits = max(np.abs(residuals.min()), residuals.max())

    n = result["n"]
    for i, variable in enumerate(variables):
        axes = figure.add_subplot(rows, 3, i + 1)

        keyed_values = sorted(zip(data[variable].values, residuals),
                              key=lambda x: x[0])
        ordered_residuals = [x[1] for x in keyed_values]

        axes.plot(list(range(0, n)), ordered_residuals, '.', color="dimgray",
                  alpha=0.75)
        axes.axhline(y=0.0, xmin=0, xmax=n, c="firebrick", alpha=0.5)
        axes.set_ylim((-limits, limits))
        axes.set_ylabel("residuals")
        axes.set_xlabel(variable)

    plt.show()
    plt.close()

    return residuals

def data_collection():
    result = dict()
    result[ "train"] = defaultdict( list)
    result[ "test"] = defaultdict( list)
    return result

def resample(data):
    n = len(data)
    return [data[ i] for i in [stats.randint.rvs(0, n - 1) for _ in range( 0, n)]]

def chunk(xs, n):
    k, m = divmod(len(xs), n)
    return [xs[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]

def cross_validation(algorithm, formula, data, evaluate, fold_count=10, repetitions=1):
    indices = list(range(len( data)))
    metrics = []
    for _ in range(repetitions):
        random.shuffle(indices)
        folds = chunk(indices, fold_count)
        for fold in folds:
            test_data = data.iloc[fold]
            train_indices = [idx not in fold for idx in indices]
            train_data = data.iloc[train_indices]
            result = algorithm(formula, data=train_data)
            model = result["model"]
            y, X = patsy.dmatrices(formula, test_data, return_type="matrix")
            # y = np.ravel( y) # might need for logistic regression
            results = summarize(formula, X, y, model)
            metric = evaluate(results)
            metrics.append(metric)
    return metrics


def sse(results):
    errors = results['residuals']
    n = len( errors)
    squared_error = np.sum( [e**2 for e in errors])
    return (np.sqrt((1.0/n) * squared_error), results['r_squared'], results['sigma'])


def learning_curves(algorithm, formula, data, evaluate, fold_count=10,
                    repetitions=3, increment=1):
    indices = list(range(len(data)))
    results = data_collection()
    for _ in range(repetitions):
        random.shuffle(indices)
        folds = chunk(indices, fold_count)
        for fold in folds:
            test_data = data.iloc[fold]
            train_indices = [idx for idx in indices if idx not in fold]
            train_data = data.iloc[train_indices]
            for i in list(range(increment, 100, increment)) + [
                100]:  # ensures 100% is always picked.
                # the indices are already shuffled so we only need to take ever increasing chunks
                train_chunk_size = int(np.ceil((i / 100) * len(train_indices)))
                train_data_chunk = data.iloc[train_indices[0:train_chunk_size]]
                # we calculate the model
                result = algorithm(formula, data=train_data_chunk)
                model = result["model"]
                # we calculate the results for the training data subset
                y, X = patsy.dmatrices(formula, train_data_chunk,
                                       return_type="matrix")
                result = summarize(formula, X, y, model)
                metric = evaluate(result)
                results["train"][i].append(metric)

                # we calculate the results for the test data.
                y, X = patsy.dmatrices(formula, test_data,
                                       return_type="matrix")
                result = summarize(formula, X, y, model)
                metric = evaluate(result)
                results["test"][i].append(metric)
            #
        #
    # process results
    # Rely on the CLT...
    statistics = {}
    for k, v in results["train"].items():
        statistics[k] = (np.mean(v), np.std(v))
    results["train"] = statistics
    statistics = {}
    for k, v in results["test"].items():
        statistics[k] = (np.mean(v), np.std(v))
    results["test"] = statistics
    return results


def results_to_curves( curve, results):
    all_statistics = results[ curve]
    keys = list( all_statistics.keys())
    keys.sort()
    mean = []
    upper = []
    lower = []
    for k in keys:
        m, s = all_statistics[ k]
        mean.append( m)
        upper.append( m + 2 * s)
        lower.append( m - 2 * s)
    return keys, lower, mean, upper


def plot_learning_curves(results, metric, desired=None, zoom=False,
                         credible=True):
    figure = plt.figure(figsize=(10, 6))

    axes = figure.add_subplot(1, 1, 1)

    xs, train_lower, train_mean, train_upper = results_to_curves("train",
                                                                 results)
    _, test_lower, test_mean, test_upper = results_to_curves("test", results)

    axes.plot(xs, train_mean, color="steelblue", label="train")
    axes.plot(xs, test_mean, color="firebrick", label="test")
    if credible:
        axes.fill_between(xs, train_upper, train_lower, color="steelblue",
                          alpha=0.25)
        axes.fill_between(xs, test_upper, test_lower, color="firebrick",
                          alpha=0.25)

    if desired:
        if type(desired) is tuple:
            axes.axhline((desired[0] + desired[1]) / 2.0, color="gold",
                         label="desired")
            axes.fill_between(xs, desired[1], desired[0], color="gold",
                              alpha=0.25)
        else:
            axes.axhline(desired, color="gold", label="desired")

    axes.legend()
    axes.set_xlabel("training set (%)")
    axes.set_ylabel(metric)
    axes.set_title("Learning Curves")

    if zoom:
        y_lower = int(0.9 * np.amin([train_lower[-1], test_lower[-1]]))
        y_upper = int(1.1 * np.amax([train_upper[-1], test_upper[-1]]))
        axes.set_ylim((y_lower, y_upper))

    plt.show()
    plt.close()
#


def x_val(X, Y, model, fold_count=10, reps=3):
    (n_data, _) = X.shape
    ind = np.arange(n_data) % fold_count
    r2 = {}
    r2['train'] = []
    r2['test'] = []

    result = {}

    for rep in np.arange(reps):
        np.random.shuffle(ind)
        x_test = X[ind == 0]
        x_train = X[ind != 0]
        y_test = Y[ind == 0]
        y_train = Y[ind != 0]

        n_test = np.sum(ind == 0)
        n_train = np.sum(ind != 0)

        model.fit(x_train, y_train)
        yhat_train = model.predict(x_train)
        res_train = y_train / (n_train ** (1 / 2)) - yhat_train / (
                    n_train ** (1 / 2))
        mse_train = np.sum(res_train ** 2)
        r2_train = 1 - mse_train / np.var(y_train)

        yhat_test = model.predict(x_test)
        res_test = y_test / (n_test ** (1 / 2)) - yhat_test / (
                    n_test ** (1 / 2))
        mse_test = np.sum(res_test ** 2)
        r2_test = 1 - mse_test / np.var(y_test)
        # print("1-{:.3f}/{:.3f}={:.3f}".format(mse_test, np.var(y_test), r2_test) )

        r2['train'].append(r2_train / reps)
        r2['test'].append(r2_test / reps)

    result['train'] = np.sum(r2['train'])
    result['test'] = np.sum(r2['test'])

    return result


def val_cur(X, Y, model, param, ks, fold_count=10, reps=3):
    result = {'train': [],
              'test': []}

    for k in ks:
        d = {param: k}
        model = model.set_params(**d)
        temp = x_val(X, Y, model)
        result['train'].append(temp['train'])
        result['test'].append(temp['test'])

    return result


def boot(X, Y, F, n=100):
    result = {'train': np.array([]),
              'test': np.array([])}

    (n_data, _) = X.shape

    for ind in np.arange(n):
        boot_ind = np.random.choice(np.arange(n_data),
                                    n_data, replace=True)
        X_boot = X.loc[boot_ind]
        Y_boot = Y.loc[boot_ind]
        temp = F(X_boot, Y_boot)
        if result['train'].size == 0:
            result['train'] = np.array(temp['train'])
            result['test'] = np.array(temp['test'])

        else:
            result['train'] = np.vstack((result['train'],
                                         np.array(temp['train'])))
            result['test'] = np.vstack((result['test'],
                                        np.array(temp['test'])))

    return result

