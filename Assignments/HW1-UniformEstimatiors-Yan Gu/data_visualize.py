import matplotlib.pyplot as plt
import pandas as pd

def show_Picture(x_data, y_data_1, y_data_2, y_data_name1, y_data_name2, x_label, y_label, title):
    plt.figure(figsize=(16, 8))

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    plt.plot(x_data, y_data_1, c='red', lw=0.5, label=y_data_name1)
    plt.plot(x_data, y_data_2, c='blue', lw=0.5, label=y_data_name2)

    plt.legend(loc='upper left')

    filename = 'Figure' + title[4] + '.png'

    print(title[4])

    # save the picture，filename is title
    plt.savefig(filename, bbox_inches='tight')

    plt.show()


if __name__ == "__main__":
    data = pd.read_csv('result.csv')

    col_j = data['j']
    L_MOM = data['L_MOM']
    L_MLE = data['L_MLE']
    MSE_MOM = data['MSE_MOM']
    MSE_MLE = data['MSE_MLE']
    MSE_MOM_T = data['MSE_MOM_T']
    MSE_MLE_T = data['MSE_MLE_T']

    show_Picture(col_j, L_MOM, L_MLE, "L_bar_MOM", "L_bar_MLE", "j", "L_bar", 
                "Fig 1: Comparison between L_bar_MOM and L_bar_MLE for each j iteration.")

    show_Picture(col_j, MSE_MOM, MSE_MLE, "MSE for MOM", "MSE for MLE", "j", "MSE value", 
                "Fig 2: Comparison of MSE between MOM and MLE for each j iteration.")

    show_Picture(col_j, MSE_MOM, MSE_MOM_T, "estimated MSE", "theoretical MSE", "j", "MSE value", 
                "Fig 3: Comparison of MSE between estimated value and theoretical value for each j iteration.")

    show_Picture(col_j, MSE_MLE, MSE_MLE_T, "estimated MSE", "theoretical MSE", "j", "MSE value", 
                "Fig 4: Comparison of MSE between estimated value and theoretical value for each j iteration.")
