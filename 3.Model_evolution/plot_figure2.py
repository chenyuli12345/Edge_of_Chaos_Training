import os #os模块提供了许多与操作系统交互的函数
import pickle #用来在文件之间存储和读取python对象，实现了基本的数据序列化（将对象信息保存到文件中）和反序列化（从文件中创建出原来的对象）

import matplotlib #绘图工具
import matplotlib.patches as mpatches #包含了一些图形类，如原圆形，矩形，多边形等
import matplotlib.pyplot as plt #提供了matlab风格的绘图api
import numpy as np
import scipy.io as spio #scipy是一个用于科学计算和技术计算的Python库，io模块提供了一些输入/输出功能
import seaborn as sns #基于matplotlib的绘图库，提供了更多的绘图功能
from matplotlib.lines import Line2D #用于绘制线条
from matplotlib.ticker import FormatStrFormatter #用于设置坐标轴的刻度格式
from mpl_toolkits.axes_grid1.inset_locator import (inset_axes, mark_inset, zoomed_inset_axes) #导入几个函数，用于在图中创建插图和标记插图的位置
from scipy import stats #scipy是一个用于科学计算和技术计算的Python库，stats模块包含了一些统计函数
from scipy.optimize import curve_fit #用于拟合数据

#用于绘制神经网络训练过程中权重的变化路径，接受参数args，是一个包含各种超参数设置的对象
def plot_weights_path(args):
    '''Plot the path of this hyperparameter setting.
    '''
    #创建一个新的图形和子图，并准备绘图数据
    # Plot model evolution path
    # fig, ax = plt.subplots(figsize=(11, 6.6))
    fig, ax = plt.subplots(figsize=(13, 6.5)) #创建一个新的图形和子图，指定子图的行数为1，列数未指定（默认为1），整个图形的大小为13*6.5英寸
    betas, J0s, separation1_log = prepare_xy(args) #传入参数args，调用下面的prepare_xy函数，获取代表纵轴刻度的betas，代表横轴刻度的J0s，以及代表混沌度的separation1_log（横轴只有50~100的数据）
    
    
    # Plot separation，绘制热图
    ax = sns.heatmap(separation1_log, xticklabels=J0s, yticklabels=betas[::-1], cmap='Purples') #调用sns库中的heatmap函数在子图ax上绘制热图，热图数据来源于separation1_log，x轴来自于J0s，y轴来自于betas（反转顺序），颜色映射camp为紫色
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=24, rotation=0) #设置子图ax的x轴刻度标签，函数ax.get_xticklabels()的作用是获取当前x轴的刻度标签，返回一个标签列表;而ax.set_xticklabels()函数的作用是设置x轴的刻度标签，第一个参数是一个标签列表，第二个参数fontsize是字体大小为24，第三个参数rotation是旋转角度为0度，即标签会水平显示
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=24, rotation=0) #设置子图ax的y轴刻度标签，函数ax.get_yticklabels()的作用是获取当前y轴的刻度标签，返回一个标签列表;而ax.set_yticklabels()函数的作用是设置y轴的刻度标签，第一个参数是一个标签列表，第二个参数fontsize是字体大小为24，第三个参数rotation是旋转角度为0度，即标签会水平显示
    ax.set_xlabel(r'$J_0/\ J$', fontsize=24) #设置x轴标签为$J_0/\ J$，字体大小为24
    ax.set_ylabel(r'$1/\ J$', fontsize=24) #设置y轴标签为$1/\ J$，字体大小为24



    # Plot colorbar，设置颜色条以及其刻度、标签
    cbar = ax.collections[0].colorbar #获取热图的颜色条，颜色条用于表示颜色和数据值之间的对应关系
    cbar.set_label('Asymptotic distance $\|\mathbf{x}\'_\infty-\mathbf{x}_\infty\|$', fontsize=24) #设置热图颜色条的标签，字体大小为24
    if args.activation_func == 'relu': #如果激活函数为'relu'
        cbar_labels1 = [-4, -2, 0, 2] #定义一个新列表，用来设置热图颜色条的刻度位置
        cbar_labels2 = ['< {}'.format(.0001), str(0.01), str(1.0), '>= {}'.format(100)] #再定义一个新列表，包含了颜色条的刻度标签
    elif args.activation_func == 'tanh':
        cbar_labels1 = [-4, -2, 0] #定义一个新列表，用来设置热图颜色条的刻度位置
        cbar_labels2 = ['< {}'.format(.0001), str(0.01), '>= {}'.format(1)] #再定义一个新列表，包含了颜色条的刻度标签
    cbar.set_ticks(cbar_labels1) #设置热图颜色条的刻度位置，位置被设置在cbar_labels1中的数据上
    cbar.set_ticklabels(cbar_labels2) #设置热图颜色条的刻度标签
    cbar.ax.tick_params(labelsize=24) #设置热图颜色条的刻度标签的字体大小为24



    # Plot theoretical boundary，绘制理论边界
    if args.activation_func == 'relu': #如果激活函数为'relu'
        # Prepare boundary data to plot
        mat = spio.loadmat('rawdata/data_100bins_relu.mat', squeeze_me=True) #从文件'rawdata/data_100bins_relu.mat'中加载数据，该函数返回一个字典，包含了该文件中存储的所有变量，存储在mat中
        integrals = mat['integrals'][::-1] #从mat中取出键为'integrals'的值，并反转顺序，赋值给integrals
        # integrals = integrals.asty
        for i in range(0, integrals.shape[1], 2): #遍历integrals的第二个维度，步长为2
            integrals[:, i] = 0 #将integrals的第i列的所有元素都设置为0
        mask = (integrals - 1) * (-1) #将integrals中所有元素都减1，然后取负数，赋值给mask
        sns.heatmap(integrals, xticklabels=J0s, yticklabels=betas[::-1], mask=mask, cmap='gray', cbar=False) #调用sns库中的heatmap函数绘制热图，热图数据来源于integrals，x轴来自于J0s，y轴来自于betas（反转顺序），颜色映射camp为紫色
    elif args.activation_func == 'tanh': #如果激活函数为'tanh'
        plt.axhline(y=(100.5-1*33.3333), color='black', linestyle='--', lw=3) #该函数用于在图形上绘制一条水平线，y是水平线的y坐标，color是线的颜色，linestyle是线的样式，lw是线的宽度，x坐标的默认xmin和xmax是0和1，代表线覆盖了整个x轴范围
    

    
    # Plot paths，绘制模型的演化路径
    results = read_weights(args, args.weight_decay) #调用下面自定义的read_weights函数，读取权重数据
    losses = read_losses(args, args.weight_decay) #调用下面自定义的read_losses函数，读取损失数据

    #下面三个变量通过列表推导式从上面的results和losses中提取数据，并转换为numpy数组
    #创建一个二维数组W0s，元素来自于results字典。内部部分遍历args.epochs//1次(整除的意思)，每次取出字典results中键为num_repeat和epoch的值（一个字典），再从这个字典中获取'W0s'键对应的值
    W0s = np.array([[results[num_repeat][epoch]['W0s'] for epoch in range(args.epochs//1)] for num_repeat in range(1)]) 
    val_loss = np.array([[losses[num_repeat]['val'][epoch][1] for epoch in range(args.epochs//1)] for num_repeat in range(1)])
    val_acc = np.array([[losses[num_repeat]['val'][epoch][2] for epoch in range(args.epochs//1)] for num_repeat in range(1)])
    
    
    #首先计算W0s数组沿着第一个轴的平均值；然后std函数计算这个平均值数组沿着第二、第三个轴的标准差；最后这个标准差乘以神经元数量（input_shape）的平方根
    J = (np.std(W0s.mean(axis=0), axis=(1, 2))*np.sqrt(args.input_shape)) 
    #计算J0，首先计算W0s数组沿着第一个轴的平均值；然后计算这个平均值数组沿着第二、第三个轴的平均值；最后这个平均值乘以神经元数量（input_shape）
    J0 = np.mean(W0s.mean(axis=0), axis=(1, 2))*args.input_shape


    if args.activation_func == 'relu': #若激活函数为relu
        x, y = (J0/J)*20+100.5, 100.5-(1/J)*50
        plt.plot(x, y, 'o-', color='C1',
                 markersize=3.5, lw=2, label='Model evolution path')
        x_optimal = ((J0/J)*20+100.5)[np.argmin(val_loss)]
        y_optimal = (100.5-(1/J)*50)[np.argmin(val_loss)]
        plt.scatter(x_optimal, y_optimal, s=300, marker='o', color='C2',
                    label='Optimal epoch', zorder=10)
        plt.scatter(x[-1], y[-1], s=300, marker='s',
                    color='C1', label='Epoch {}'.format(args.epochs-1))
    elif args.activation_func == 'tanh': #若激活函数为tanh
        x, y = (J0/J)*50+50.5, 100.5-(1/J)*33.3333 #计算x和y的值，x的值为均值除以方差*50再加上50.5，y的值为100.5减去1除以方差*33.3333
        plt.plot(x, y, 'o-', color='C1', markersize=3, lw=2, label='Model evolution path') #绘制一个线图，其中x和y是坐标，o-表示用圆圈标记，-表示用线连接，color表示颜色，markersize表示标记大小，lw表示线宽，label表示图例
        plt.scatter(x[-1], y[-1], s=300, marker='s', color='C1', label='Epoch {}'.format(args.epochs-1)) #绘制一个散点图，其中x和y是坐标，s表示标记大小，marker表示标记形状s代表方形，color表示颜色，label表示图例
        x_optimal = ((J0/J)*50+50.5)[np.argmin(val_loss)] #计算最优点的x坐标，这里用了numpy的argmin函数，返回val_loss最小值的索引
        y_optimal = (100.5-(1/J)*33.3333)[np.argmin(val_loss)] #计算最优点的y坐标，这里用了numpy的argmin函数，返回val_loss最小值的索引
        plt.scatter(x_optimal, y_optimal, s=300, marker='o', color='C2', label='Optimal epoch (test loss)', zorder=10) #绘制一个散点图，其中x_optimal和y_optimal是坐标，s表示标记大小，marker表示标记形状o代表圆圈，color表示颜色，label表示图例，zorder表示图层顺序
    
    
    # Plot arrow，添加一个箭头来表示权重的变化方向
    u, v = np.diff(x), np.diff(y) #计算x和y的离散差分，得到适量的u和v分量，这里函数返回的是一个由相邻数组元素的差值构成的数组
    pos_x, pos_y = x[:-1] + u/2, y[:-1] + v/2 #计算箭头的起始位置pos_x和pos_y，分别为x和y的前n-1个元素加上u/2和v/2
    norm = np.sqrt(u**2+v**2) #计算矢量的长度，它是由u和v的平方和的平方根得到的
    #绘制箭头，这个函数的参数为别为：箭头的起始位置（前两个参数），箭头的方向（第三四个参数），箭头的宽度为0.002，箭头的线宽为0.5，箭头的比例为20，箭头的头部宽度为20，箭头的头部长度为20，箭头的头部轴长为16，角度时在xy平面坐标系中测量的，箭头的枢轴为mid（表示箭头的旋转点在箭头的中点），颜色为C1（表示橙色）
    plt.quiver(pos_x[0], pos_y[0], (u/norm)[0], (v/norm)[0], width=0.002, lw=0.5, scale=20, headwidth=20, headlength=20, headaxislength=16, angles="xy", pivot="mid", color='C1')
    
    # Manually add a legend，添加图例
    handles, labels = ax.get_legend_handles_labels() #调用函数获取当前图形的图例句柄和标签
    lines = Line2D([0], [0], color='black', linewidth=3, linestyle='--') #创建一个线条对象，前两个元素表示线的起始和结束位置，颜色为黑色，线宽为3，线型为虚线
    label = 'Edge of chaos ( $J=1$ )' #设置图例标签
    handles.insert(0, lines) #将线条对象插入到图例句柄的第一个位置
    labels.insert(0, label) #将图例标签插入到标签的第一个位置
    plt.legend(handles=handles, labels=labels, fontsize=20) #绘制图例，handles表示图例句柄，labels表示图例标签，fontsize表示字体大小为20
    ax.set_xlabel(r'$J_0/\ J$', fontsize=24) #设置x轴标签为$J_0/\ J$，字体大小为24
    ax.set_ylabel(r'$1/\ J$', fontsize=24) #设置y轴标签为$1/\ J$，字体大小为24


    #保存图形
    fig.tight_layout()
    if not os.path.exists('results/{}_{}'.format(args.arguments, args.num_repeats)):
        os.makedirs('results/{}_{}'.format(args.arguments, args.num_repeats))
    plt.savefig('results/{}_{}/path'.format(args.arguments,
                                            args.num_repeats), dpi=300)


def plot_weights_var_mean(args):
    '''Plot variance and mean of the weights of this hyperparameter setting.
    '''
    # plot variance
    fig, ax = plt.subplots(figsize=(6.5, 5))
    results = read_weights(args, args.weight_decay)
    losses = read_losses(args, args.weight_decay)
    W0s = np.array([[results[num_repeat][epoch]['W0s']
                     for epoch in range(args.epochs//1)] for num_repeat in range(0, 1)])
    val_loss = np.array([[losses[num_repeat]['val'][epoch][1]
                          for epoch in range(args.epochs//1)] for num_repeat in range(0, 1)])
    val_acc = np.array([[losses[num_repeat]['val'][epoch][2]
                         for epoch in range(args.epochs//1)] for num_repeat in range(0, 1)])

    J = (np.std(np.array(W0s.mean(axis=0)), axis=(1, 2))*np.sqrt(args.input_shape))
    J_squard = np.power(J, 2)
    x = np.arange(args.epochs)
    ax.plot(x[::2], J_squard[::2], 'o', markersize=3, color='C1')
    plt.scatter(x[np.argmin(val_loss.mean(axis=0))], J_squard[np.argmin(val_loss.mean(axis=0))],
                s=200, marker='o', color='C2', zorder=4, label='Optimal epoch')
    # plt.scatter(x[np.argmax(val_acc.mean(axis=0))], J_squard[np.argmax(val_acc.mean(axis=0))],
    #             s=200, marker='o', color='C3', zorder=5, label='Optimal epoch')
    # plot_lin_regress(args, J_squard)
    ax.set_xticks([0, 250, 500])
    ax.tick_params(axis='both', which='major', labelsize=24)
    # plt.ylim(0.2, 1.02)
    plt.xlabel('Epoch', fontsize=24)
    plt.ylabel(r'$J^2$', fontsize=24)
    plt.legend(fontsize=17, loc='upper left')

    axins = inset_axes(ax, width='58%', height='58%', loc='lower left',
                       bbox_to_anchor=(.38, .065, 1, 1), bbox_transform=ax.transAxes)
    # axins.axis([-4, 51, 0.2, 1.])
    # axins.plot(x[:50:2], J_squard[:50:2], 'o', markersize=4.5)
    # plot_lin_regress(args, x[:50:2], J_squard[:50:2])
    axins.axis([-4, 51, 0.2, 1.])
    axins.plot(x[:50:2], J_squard[:50:2], 'o', markersize=4.5, color='C1')
    plot_lin_regress(args, x[:50:2], J_squard[:50:2])
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    axins.yaxis.get_major_locator().set_params(nbins=3)
    axins.xaxis.get_major_locator().set_params(nbins=3)
    plt.setp(axins.get_xticklabels(), visible=False)
    plt.setp(axins.get_yticklabels(), visible=False)
    plt.legend(fontsize=13, loc='lower right')

    fig.tight_layout()
    if not os.path.exists('results/{}_{}'.format(args.arguments, args.num_repeats)):
        os.makedirs('results/{}_{}'.format(args.arguments, args.num_repeats))
    plt.savefig('results/{}_{}/variance'.format(args.arguments,
                                                args.num_repeats), dpi=300)

    # plot mean
    fig, ax = plt.subplots(figsize=(6.5, 5))
    J0 = np.mean(np.array(W0s.mean(axis=0)), axis=(1, 2))*args.input_shape
    ax.plot(J0, 'o', markersize=3)
    ax.tick_params(axis='both', which='major', labelsize=24)
    plt.xlabel('Epoch', fontsize=24)
    plt.ylabel(r'$J_0$', fontsize=24)
    plt.ylim(top=1)
    plt.legend(fontsize=22)

    fig.tight_layout()
    if not os.path.exists('results/{}_{}'.format(args.arguments, args.num_repeats)):
        os.makedirs('results/{}_{}'.format(args.arguments, args.num_repeats))
    plt.savefig('results/{}_{}/mean'.format(args.arguments,
                                            args.num_repeats), dpi=300)


def plot_loss_acc(args):
    # Plot losses
    fig, ax = plt.subplots(figsize=(6.25, 4.5))
    losses = read_losses(args, args.weight_decay)
    val_loss = np.array([[losses[num_repeat]['val'][epoch][1]
                          for epoch in range(args.epochs//1)] for num_repeat in range(1)])
    train_loss = np.array([[losses[num_repeat]['train'][epoch][1]
                            for epoch in range(args.epochs//1)] for num_repeat in range(1)])
    val_acc = np.array([[losses[num_repeat]['val'][epoch][2]
                         for epoch in range(args.epochs//1)] for num_repeat in range(1)])
    train_acc = np.array([[losses[num_repeat]['train'][epoch][2]
                           for epoch in range(args.epochs//1)] for num_repeat in range(1)])

    x = np.arange(args.epochs)
    plt.plot(x[::2], train_loss.mean(axis=0)[::2], 'o-',
             markersize=4, lw=1.5, label='Train')
    plt.scatter(x[np.argmin(val_loss.mean(axis=0))], val_loss.mean(axis=0)[np.argmin(val_loss.mean(axis=0))],
                s=300, marker='o', color='C2', zorder=5, label='Optimal epoch')  # , loss: {:.03f}'.format(val_loss.mean(axis=0)[np.argmin(val_loss.mean(axis=0))]))
    plt.plot(x[::2], val_loss.mean(axis=0)[::2], 'o-',
             markersize=4, lw=1.5, label='Test')
    plt.ylim(-0.02, 1)
    ax.set_xticks([0, 250, 500])
    # ax.set_xticklabels([])
    ax.set_yticks([0, 0.4, 0.8])
    ax.tick_params(axis='both', which='major', labelsize=24)
    plt.xlabel('Epoch', fontsize=24)
    plt.ylabel('Loss', fontsize=24)
    plt.legend(fontsize=19, loc='lower left', bbox_to_anchor=(0.35, 0.05))

    fig.tight_layout()
    if not os.path.exists('results/{}_{}'.format(args.arguments, args.num_repeats)):
        os.makedirs('results/{}_{}'.format(args.arguments, args.num_repeats))
    plt.savefig('results/{}_{}/loss'.format(args.arguments,
                                            args.num_repeats), dpi=300)

    # Plot accuracy
    fig, ax = plt.subplots(figsize=(6.25, 4.5))
    x = np.arange(args.epochs)
    plt.plot(x[::2], train_acc.mean(axis=0)[::2], 'o-',
             markersize=4, lw=1.5, label='Train')
    plt.plot(x[::2], val_acc.mean(axis=0)[::2], 'o-',
             markersize=4, lw=1.5, label='Test')
    # plt.scatter(x[np.argmax(val_acc.mean(axis=0))], val_acc.mean(axis=0)[np.argmax(val_acc.mean(axis=0))],
    #             s=300, marker='o', color='C2', zorder=5)  # , label='Optimal epoch')
    plt.scatter(x[np.argmin(val_loss.mean(axis=0))], val_acc.mean(axis=0)[np.argmin(val_loss.mean(axis=0))],
                s=300, marker='o', color='C2', zorder=5, label='Optimal epoch')
    plt.ylim(0.7, 1.01)
    ax.set_xticks([0, 250, 500])
    ax.tick_params(axis='both', which='major', labelsize=24)
    plt.xlabel('Epoch', fontsize=24)
    plt.ylabel('Accuracy', fontsize=24)
    plt.legend(fontsize=20)

    fig.tight_layout()
    if not os.path.exists('results/{}_{}'.format(args.arguments, args.num_repeats)):
        os.makedirs('results/{}_{}'.format(args.arguments, args.num_repeats))
    plt.savefig('results/{}_{}/acc'.format(args.arguments,
                                           args.num_repeats), dpi=300)


def plot_lin_regress(args, x, y):
    x_data = x
    y_data = y
    for end_epoch in [200]:
        x = x_data[:end_epoch]
        y = y_data[:end_epoch]

        # popt, pcov = curve_fit(func_0, x, y)
        # perr = np.sqrt(np.diag(pcov))
        # plt.plot(x, func_0(x, *popt), '-', linewidth=1.25, color='C3',
        #          label='Fit: ' + r'$A\times$' + 'Epoch + ' + r'$C$')

        popt, pcov = curve_fit(func_0, x, y)
        perr = np.sqrt(np.diag(pcov))
        plt.plot(x, func_0(x, *popt), '-', linewidth=1.25, color='C3',
                 label='Slope: {:.7f}'.format(popt[0]))


def func_0(x, a, b):
    return a * x + b

#用来准备x和y轴刻度值，以及预处理之前的三维混沌度数据（只取其中一部分）
def prepare_xy(args): #准备绘图数据，接受参数args，是一个包含各种超参数设置的对象
    '''
    '''
    num_bins = 100

    if args.activation_func == 'relu': #如果激活函数为'relu'
        # Prepare xy axis，准备x和y轴
        betas = [i * 2 / num_bins for i in range(1, num_bins+1)] #创建一个数组betas，表示0.02，0.04，...，1.98，2.00，共100个元素(将0~2分为100份)
        J0s = [i * 5 / num_bins for i in range(-num_bins, 21)] #创建一个数组J0s，表示-5.0，-4.95，...，0.95，1.0，共120个元素（将-5~1分为120份）
        betas = ['' for i in range(19)] + [str(beta) if i % 20 == 0 else '' for i, beta in enumerate(betas[19:])] #将betas改为字符串，且只每20个数保留数字形式的字符串
        J0s = [str(J0) if i % 24 == 0 else '' for i, J0 in enumerate(J0s)] #将J0s改为字符串，且只每24个数保留数字形式的字符串


        # Prepare data to plot，准备绘图数据
        #加载文件'rawdata/100_100_100_relu'中的数据为results1
        fname = 'rawdata/100_100_100_relu'
        with open(fname, 'rb') as f:
            results1 = pickle.load(f)
        #
        separation1 = results1[::-1, :, 0] #取出results1中的第一个维度所有元素并反转顺序、第二个维度所有元素、第三个维度的第一个元素，最后将这个三维数组赋值给separation1
        separation1[separation1 > 100] = 100 #将数组separation1中大于100的数值改为100
        separation1_log = np.log10(np.abs(separation1)+0.0001) #计算数组separation1绝对值的对数值（加上一个很小的数值，避免出现0）

    elif args.activation_func == 'tanh': #如果激活函数为'tanh'
        # Prepare xy axis，准备x轴和y轴
        betas = [i * 3 / num_bins for i in range(1, num_bins+1)] #创建一个数组betas，表示0.03，0.06，...，2.97，3.00，共100个元素(将0~3分为100份)
        J0s = [i * 2 / num_bins for i in range(-50, 51)] #创建一个数组J0s，表示-1.0，-0.98，...，0.98，1.0，共100个元素（将-1~1分为100份）
        betas = ['' for i in range(19)] + [str(beta) if i % 20 == 0 else '' for i, beta in enumerate(betas[19:])] #将betas改为字符串，且只每20个数保留数字形式的字符串
        J0s = [str(J0) if i % 20 == 0 else '' for i, J0 in enumerate(J0s)] #将J0s改为字符串，且只每20个数保留数字形式的字符串
        
        
        # Prepare data to plot，准备绘图数据
        #加载文件'rawdata/50_100_100_tanh'中的数据为results1
        fname = 'rawdata/50_100_100_tanh'
        with open(fname, 'rb') as f:
            results1 = pickle.load(f)

        separation1 = results1[::-1, :, 0] #取出results1中的第一个维度所有元素并反转顺序、第二个维度所有元素、第三个维度的第一个元素，最后将这个三维数组赋值给separation1
        separation1[separation1 > 1] = 1 #将数组separation1中大于1的数值改为1
        separation1_log = np.log10(np.abs(separation1)+0.0001)[:, 50:151] #计算数组separation1绝对值的对数值（加上一个很小的数值，避免出现0），并取出第二维度的50~150个元素，也就是横轴取第50~150个元素

    return betas, J0s, separation1_log #返回代表纵轴的betas，代表横轴的J0s，以及代表混沌度的separation1_log（横轴只有50~100的数据）


def read_weights(args, weight_decay): #接受args和权重衰减值
    """Read the files contains weights and weights difference.

    Args:
        args: configuration options dictionary

    Returns:
        weights and weights difference
    """
    results = {} #创建一个空字典
    for num_repeat in range(args.num_repeats): #遍历args.num_repeats次
        results[num_repeat] = {} #为字典results添加一个新的空字典
        #创造一个字符串，包含了数据集、优化器、激活函数、训练轮数、学习率、批大小、动量、beta_1、beta_2、权重衰减值
        arguments = '{}_{}_{}/{}_{}_{}_{}_{}_{}_{}'.format(args.dataset, args.optimizer, args.activation_func, args.epochs, int(args.lr*10000), int(
            args.batch_size), int(args.momentum*1000), int(args.beta_1*1000), int(args.beta_2*10000000), int(weight_decay*1000000))
        
        for epoch in args.log_epochs: #循环指定的epoch次
            #对于每个epoch
            fname = 'rawdata/weights/{}/{}/epoch{:05d}'.format(arguments, num_repeat, epoch) #构造一个文件名
            with open(fname, 'rb') as f: #打开这个fname命名的文件
                d = pickle.load(f) #将文件内容赋值给d
            results[num_repeat][epoch] = d #将读取的内容d存储到results字典的对应位置

    return results


def read_losses(args, weight_decay): #接受args和权重衰减值
    """Get the losses from files

    Args:
        args: configuration options dictionary

    Returns:
        loss and accuracy
    """
    losses = {} #创建一个空字典
    for num_repeat in range(args.num_repeats): #遍历args.num_repeats次
        #创建一个字符串，包含了数据集、优化器、激活函数、训练轮数、学习率、批大小、动量、beta_1、beta_2、权重衰减值
        arguments = '{}_{}_{}/{}_{}_{}_{}_{}_{}_{}'.format(args.dataset, args.optimizer, args.activation_func, args.epochs, int(args.lr*10000), int(
            args.batch_size), int(args.momentum*1000), int(args.beta_1*1000), int(args.beta_2*10000000), int(weight_decay*1000000))
        fname = 'rawdata/losses/{}/{}/losses'.format(arguments, num_repeat) #构造一个文件名
        with open(fname, 'rb') as f: #打开这个fname命名的文件
            d = pickle.load(f) #将文件内容赋值给d
        losses[num_repeat] = d  #将读取的内容d存储到losses字典的对应位置

    return losses


def read_scores(args, weight_decay): #接受args和权重衰减值
    """Get the scores from files

    Args:
        args: configuration options dictionary

    Returns:
        loss and accuracy
    """
    scores = {} #创建一个空字典
    for num_repeat in range(args.num_repeats): #遍历args.num_repeats次
        #创建一个字符串，包含了数据集、优化器、激活函数、训练轮数、学习率、批大小、动量、beta_1、beta_2、权重衰减值
        arguments = '{}_{}_{}/{}_{}_{}_{}_{}_{}_{}'.format(args.dataset, args.optimizer, args.activation_func, args.epochs, int(args.lr*10000), int(
            args.batch_size), int(args.momentum*1000), int(args.beta_1*1000), int(args.beta_2*10000000), int(weight_decay*1000000))
        fname = 'rawdata/scores/{}/{}/scores'.format(arguments, num_repeat) #构造一个文件名
        with open(fname, 'rb') as f: #打开这个fname命名的文件
            d = pickle.load(f) #将文件内容赋值给d
        scores[num_repeat] = d #将读取的内容d存储到scores字典的对应位置

    return scores
