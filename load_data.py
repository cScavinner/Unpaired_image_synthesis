import glob
import os
import torchio as tio
import sys

def load_data(data, segmentation, dynamic_path = None, static_path = None, seg_path=None, footmask_path=None):
    '''
        This function returns a list of tio.Subjects to be used for the training, validation, and test phases. Each tio.Subject contains four items: 
            - the subject identifier (for test/train/validation partitioning)
            - an LR image
            - an HR image
            - a segmentation mask (of the foot, for example) 
        The mask is used in the sampler to constrain patch extraction to the foot only and not to the background. The utilization of patches necessitates 
        the use of partially registered data, ensuring that corresponding patches contain approximately similar structures (e.g., a foot LR patch and a foot 
        HR patch instead of a foot LR patch and a background HR patch). The necessity for registration is negated when learning from full images instead of patches. 
    '''
    subjects=[]
    subject=tio.Subject(
        subject_name=s,
        LR_image=tio.ScalarImage(LR),
        HR_image=tio.ScalarImage(HR),
        label=tio.LabelMap(SEG)
    )
    subjects.append(subject)
    return(subjects, check_subjects)
