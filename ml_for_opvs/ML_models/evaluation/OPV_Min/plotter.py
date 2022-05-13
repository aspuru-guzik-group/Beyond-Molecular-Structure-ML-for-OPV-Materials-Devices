import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


def parity_plot(y_test, yhat, y_max):
    """Creates a parity plot for two column data (predicted data and ground truth data)"""
    y_test = y_test
    yhat = yhat
    # find slope and y-int of linear line of best fit
    m, b = np.polyfit(y_test, yhat, 1,)
    print(m, b)
    # find correlation coefficient
    corr_coef = np.corrcoef(y_test, yhat,)[0, 1]
    # find rmse
    rmse = np.sqrt(mean_squared_error(y_test, yhat,))

    fig, ax = plt.subplots()
    # ax.set_title("Predicted vs. Experimental PCE (%)", fontsize=22)
    ax.set_xlabel("Experimental_PCE_(%)", fontsize=18)
    ax.set_ylabel("Predicted_PCE_(%)", fontsize=18)
    ax.scatter(
        y_test, yhat, s=4, alpha=0.7, color="#0AB68B",
    )
    ax.plot(
        y_test,
        m * y_test + b,
        color="black",
        label=[
            "R: " + str(round(corr_coef, 3)) + "  " + "RMSE: " + str(round(rmse, 3))
        ],
    )
    ax.plot([0, y_max], [0, y_max], "--", color="blue", label="Perfect Correlation")
    ax.legend(loc="upper left", fontsize=14)
    ax.tick_params(axis="both", which="major", labelsize=14)
    # add text box with slope and y-int
    # textstr = "slope: " + str(m) + "\n" + "y-int: " + str(b)
    # ax.text(0.5, 0.5, textstr, wrap=True, verticalalignment="top")
    plt.show()
