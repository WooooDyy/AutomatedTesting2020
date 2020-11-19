import csv
import matplotlib.pyplot as plt


#accuracy
from matplotlib.ticker import MultipleLocator, FuncFormatter

plt.rcParams.update({'font.size': 5})

def to_percent(temp, position):
    return '%1.0f'%(100*temp) + '%'

with open("../../Data/cifar100_tables/cifar100_accuracy.csv",'r') as f:
    reader = csv.reader(f)
    aug_policy_list = [
        "crop",
        "shift",
        "rotate",
        "fliplr",
        "flipud",
        "Gaussian_noise",
        "brightness",
        "contrast",
        "crop_rotate_brightness",
        "shift_noise"
    ]
    for row in reader:
        if( row[0]=='') :
            continue
        x = range(len(aug_policy_list))
        y = [float(i) for i in row[1:]]
        # plt.legend(loc='lower right')
        plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))

        plt.plot(x, y, 'o-',label = row[0])
        plt.tick_params(axis='both', which='major', labelsize=10)

        plt.xticks(x,aug_policy_list,rotation  = 60)

        plt.legend(loc="lower right")
    plt.show()

#accuracy_minus

with open("../../Data/cifar100_tables/cifar100_accuracy_minus.csv",'r') as f:
    reader = csv.reader(f)
    aug_policy_list = [
        "crop",
        "shift",
        "rotate",
        "fliplr",
        "flipud",
        "Gaussian_noise",
        "brightness",
        "contrast",
        "crop_rotate_brightness",
        "shift_noise"
    ]
    for row in reader:
        if( row[0]=='') :
            continue
        x = range(len(aug_policy_list))
        y = [-float(i) for i in row[1:]]
        plt.legend(loc='upper right')
        plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))

        plt.plot(x, y, 'o-',label = row[0])
        plt.tick_params(axis='both', which='major', labelsize=10)

        plt.xticks(x,aug_policy_list,rotation  = 60)

        plt.legend()
    plt.show()


with open("../../Data/cifar100_tables/cifar100_accuracy_loss_rate.csv",'r') as f:
    reader = csv.reader(f)
    aug_policy_list = [
        "crop",
        "shift",
        "rotate",
        "fliplr",
        "flipud",
        "Gaussian_noise",
        "brightness",
        "contrast",
        "crop_rotate_brightness",
        "shift_noise"
    ]
    for row in reader:
        if( row[0]=='') :
            continue
        x = range(len(aug_policy_list))
        y = [-float(i) for i in row[1:]]
        # ax = plt.gca()
        # ax.xaxis.set_major_locator(MultipleLocator(100))
        plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))

        plt.plot(x, y, 'o-',label = row[0])

        plt.tick_params(axis='both', which='major', labelsize=10)
        plt.xticks(x,aug_policy_list,rotation  = 60)

        plt.legend()

    plt.show()
