import seaborn as sns
import matplotlib.pyplot as plt

def plot_boxplot_violin (data, title, min_=-1, max_=-1):
    plt.rcParams["figure.figsize"] = (16,4)
    f, axes = plt.subplots(1, 2)

    g1 = sns.violinplot(data=data, ax=axes[0], orient='h')
    g1.set(title=title)
    g2 = sns.boxplot(data=data, ax=axes[1], orient='h')
    g2.set(title=title)
    
    if(min_ != max_):
        g1.set(xlim=(min_, max_))
        g2.set(xlim=(min_, max_))

    plt.show()
    
def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        return '{v:d}\n({p:.2f}%)'.format(p=pct,v=val)
    return my_autopct