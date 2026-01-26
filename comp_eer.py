import numpy as np
import pandas as pd
import argparse

# Core functions to compute FAR, FRR, and EER
#
# From ASVspoof official package https://www.asvspoof.org/resources/tDCF_python_v2.zip
def compute_det_curve(target_scores, nontarget_scores):
    """ frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)
    
    input
    -----
      target_scores:    np.array, score of target (or positive, bonafide) trials
      nontarget_scores: np.array, score of non-target (or negative, spoofed) trials
      
    output
    ------
      frr:         np.array,  false rejection rates measured at multiple thresholds
      far:         np.array,  false acceptance rates measured at multiple thresholds
      thresholds:  np.array,  thresholds used to compute frr and far

    frr, far, thresholds have same shape = len(target_scores) + len(nontarget_scores) + 1
    """
    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate((np.ones(target_scores.size),
                             np.zeros(nontarget_scores.size)))

    # Sort labels based on scores                                                         
    indices = np.argsort(all_scores, kind='mergesort')
    labels = labels[indices]

    # Compute false rejection and false acceptance rates                                  
    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = (nontarget_scores.size -
                            (np.arange(1, n_scores + 1) - tar_trial_sums))

    frr = np.concatenate((np.atleast_1d(0), tar_trial_sums/target_scores.size))
    # false rejection rates                                                               
    far = np.concatenate((np.atleast_1d(1),
                          nontarget_trial_sums / nontarget_scores.size))
    # false acceptance rates                                                              
    thresholds = np.concatenate((np.atleast_1d(all_scores[indices[0]] - 0.001),
                                 all_scores[indices]))
    # Thresholds are the sorted scores                                                    
    return frr, far, thresholds



def compute_eer(target_scores, nontarget_scores):
    """ eer, eer_threshold = compute_det_curve(target_scores, nontarget_scores)
    
    input
    -----
      target_scores:    np.array, score of target (or positive, bonafide) trials
      nontarget_scores: np.array, score of non-target (or negative, spoofed) trials
      
    output
    ------
      eer:              scalar,  value of EER
      eer_threshold:    scalar,  value of threshold corresponding to EER
    """
    frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((frr[min_index], far[min_index]))
    return eer, thresholds[min_index]



def compute_eer_API(score_file, protocol_file):
    """eer = compute_eer_API(score_file, protocol_file)
    
    input
    -----
      score_file:     string, path to the socre file
      protocol_file:  string, path to the protocol file
    
    output
    ------
      eer:  scalar, eer value
      
    The way to load text files using read_csv depends on the text format.
    Please change the read_csv if necessary
    """
    # load score
    score_pd = pd.read_csv(score_file, sep = ' ', names = ['trial', 'score'], index_col = 'trial', skipinitialspace=True)
    # load protocol
    protocol_pd = pd.read_csv(protocol_file, sep = ' ', names = ['speaker', 'trial', '-', 'attack', 'label'], index_col = 'trial')
    # joint together
    merged_pd = score_pd.join(protocol_pd)
    #merge
    #
    bonafide_scores = merged_pd.query('label == "bonafide"')['score'].to_numpy()
    spoof_scores = merged_pd.query('label == "spoof"')['score'].to_numpy()
    
    eer, th = compute_eer(bonafide_scores, spoof_scores)
    # print("th", th)
    return eer, th

def compute_accuracy(score_file, protocol_file, threshold):
    # load score
    score_pd = pd.read_csv(score_file, sep = ' ', names = ['trial', 'score'], index_col = 'trial', skipinitialspace=True)
    # load protocol
    protocol_pd = pd.read_csv(protocol_file, sep = ' ', names = ['speaker', 'trial', '-', 'attack', 'label'], index_col = 'trial')
    # joint together
    merged_pd = score_pd.join(protocol_pd)
    #merge
    bonafide_scores = merged_pd.query('label == "bonafide"')['score'].to_numpy()
    spoof_scores = merged_pd.query('label == "spoof"')['score'].to_numpy()
    TP = np.count_nonzero(bonafide_scores>=threshold)
    TN = np.count_nonzero(spoof_scores<threshold)
    FP = np.count_nonzero(spoof_scores>=threshold)
    FN = np.count_nonzero(bonafide_scores<threshold)
    all_score_size = bonafide_scores.size + spoof_scores.size

    accuracy = (TN+TP)/all_score_size
    recall = TP/(TP+FN)
    return accuracy , recall

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--protocol', type=str,
                        help='path of protocol file')
    parser.add_argument('-s', '--score', type=str,
                        help='path of score file')
    args = parser.parse_args()
    prot = args.protocol
    score = args.score

    print("Protocol: ", prot)
    print("Score: ", score)
    eer, th = compute_eer_API(score, prot)
    accuracy, recall = compute_accuracy(score, prot, th)
    print("EER (%): {:.4f}".format(eer * 100))
    print("Threshold : {:.8f}".format(th))
    print("accuracy (%): {:.4f}".format(accuracy * 100))
    print("recall (%): {:.4f}\n".format(recall * 100))
  