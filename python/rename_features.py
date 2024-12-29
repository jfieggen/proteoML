def rename_features(feature_names):
    print("Using the updated rename_features function")
    renamed_features = []
    for feature in feature_names:
        if feature == 'BaC_Sex':
            renamed_features.append('Sex')
        elif feature == 'BaC_Sex_Female':
            renamed_features.append('Sex (Female)')
        elif feature == 'BaC_Sex_Male':
            renamed_features.append('Sex (Male)')
        elif feature == 'TEU_BaC_AgeAtRec':
            renamed_features.append('Age')
        elif feature == 'waist_hip_ratio_binary':
            renamed_features.append('Elevated Waist-Hip Ratio')
        elif feature == 'TEU_ethnicgrp':
            renamed_features.append('Ethnic Group')
        elif feature == 'BlA_Haemoglob':
            renamed_features.append('Haemoglobin')
        elif feature == 'BlA_PlateCount':
            renamed_features.append('Platelet Count')
        elif feature == 'BlA_WhiteBCCount':
            renamed_features.append('White Blood Cell Count')
        elif feature == 'BlA_MCVol':
            renamed_features.append('MCV')
        elif feature == 'hemoglobin_binary':
            renamed_features.append('Hemaaoglobin (Binary)')
        elif feature == 'anaemia':
            renamed_features.append('Anemia')
        elif feature == 'HMH_PainBack':
            renamed_features.append('Back Pain')
        elif feature == 'HMH_ChestPain':
            renamed_features.append('Chest Pain')
        elif feature == 'BBC_CRP_Result':
            renamed_features.append('CRP')
        elif feature == 'BBC_CA_Result':
            renamed_features.append('Calcium')
        elif feature == 'BBC_TP_Result':
            renamed_features.append('TP Result')
        elif feature == 'BBC_ALB_Result':
            renamed_features.append('Albumin Result')
        elif feature == 'PhA_METsWkAllAct':
            renamed_features.append('METs per Week (All Activities)')
        elif feature == 'TEU_Smo_Status':
            renamed_features.append('Smoking Status')
        elif feature == 'TEU_Alc_Status':
            renamed_features.append('Alcohol Status')
        elif feature == 'BBC_CHOL_Result':
            renamed_features.append('Cholesterol Result')
        elif feature == 'C10AA':
            renamed_features.append('C10AA')
        elif feature == 'BlA_MReticVol':
            renamed_features.append('Mean Reticulocyte Volume')
        elif feature == 'BBC_CYS_Result':
            renamed_features.append('Cystatin C Result')
        elif feature == 'BSM_BMI':
            renamed_features.append('BMI')
        elif feature == 'Imp_Body':
            renamed_features.append('Impedance Body')
        elif feature == 'Imp_TrunkPredMass':
            renamed_features.append('Impedance Trunk Predicted Mass')
        elif feature == 'BBC_TES_Result':
            renamed_features.append('Testosterone Result')
        elif feature == 'TEU_HES_K40_prev':
            renamed_features.append('K40 Previous')
        elif feature == 'TEU_HES_D89_prev':
            renamed_features.append('D89 Previous')
        elif feature == 'TEU_CaR_D47_prev':
            renamed_features.append('D47 Previous')
        elif feature == 'BlA_LymphoPC':
            renamed_features.append('Lymphocyte Percent Count')
        elif feature == 'BlA_WhiteBCCount':
            renamed_features.append('White Blood Cell Count')
        elif feature == 'BlA_RedBCCount':
            renamed_features.append('Red Blood Cell Count')
        elif feature == 'BBC_APOA_Result':
            renamed_features.append('Apolipoprotein A Result')
        elif feature.startswith('HLA.A.'):
            renamed_features.append('HLA.A')
        elif feature.startswith('HLA.DRA.'):
            renamed_features.append('HLA.DRA')
        elif feature.startswith('HLA.E.'):
            renamed_features.append('HLA.E')
        else:
            renamed_features.append(feature.split('.')[0])
    return renamed_features