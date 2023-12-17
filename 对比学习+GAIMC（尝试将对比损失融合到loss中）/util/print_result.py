from util.cluster import cluster
import warnings

warnings.filterwarnings('ignore')


def print_result(n_clusters, H, gt, count=10): # gt就是ground true
    acc_avg, acc_std, nmi_avg, nmi_std, ri_avg, ri_std, f1_avg, f1_std,pur = cluster(n_clusters, H, gt, count=count)
    print('clustering h      : acc = {:.4f}, nmi = {:.4f}'.format(acc_avg, nmi_avg))
    print(f'acc_std:{acc_std}')
    return acc_avg,nmi_avg,ri_avg,f1_avg,pur
