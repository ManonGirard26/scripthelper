"""
             IMPORTANT

            Si on fait une modification dans ce script, il fait ABSOLUMENT l'exécuter
            pour le déployer dans les différents répertoires des steps de pipeline.
            La raison?  Un snapshot sera pris lors du lancement d'un step et question
            d'efficacité et réutilisation de code on copie une instance de ce script utilitaire
            pour que sur l'image docker, le script du pipeline puisse l'utiliser.
"""
import os
import glob
import requests
import argparse
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import pyodbc
import pickle
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn import metrics
from time import time, sleep
from pathlib import Path
from azureml.data import DataType
from dotenv import load_dotenv
from stat import S_IREAD
from stat import S_IWRITE
from azureml.core import Workspace
from azureml.core import Dataset
from azureml.core import Datastore
from azureml.core import Run
from azureml.core.compute import AmlCompute, ComputeTarget
from azureml.core.compute_target import ComputeTargetException
from azureml.core.authentication import AzureCliAuthentication
from azureml.core.authentication import InteractiveLoginAuthentication
from azureml.pipeline.core import PipelineEndpoint
from azureml.pipeline.core import PipelineRun
from azureml.core.experiment import Experiment
from msrest.exceptions import HttpOperationError
from azureml.exceptions._azureml_exception import UserErrorException
from azureml.core import Environment
from azureml.data.datapath import DataPath


class MLOPSCONST(object):
    """ Constantes pour établir nomenclature """
    RAWTYPE = "raw"
    RAWREMOVEIMPUTETYPE = "raw_remove_impute"
    LABEL = "label"
    PREPAREDTYPE = "prepared"
    DATATYPE_TRAIN = "train"
    DATATYPE_TEST = "test"
    DATATYPE_SUBSET_TRAIN = "subset_train"
    DATATYPE_SUBSET_TEST = "subset_test"
    DATATYPE_SUBSET_TEST_FAIRNESS = "subset_test_fairness"
    DOCKER_SPECIFICATIONS_FILE = "DockerFileODBCSql"
    DOCKER_SPECIFICATIONS_IMAGE_BASE_FILE = "azureml-images/edo_nomad"
    CONDA_SPECIFICATIONS_YML = "conda_dependencies.yml"
    CONDA_ENV_FROM_EXISTING = "condafromexisting"
    CONDA_ENV_FROM_FILE = "condafromymlfile"
    COMPUTE_TARGET_LOCAL = "local"
    COMPUTE_TARGET_CLOUD = "cloud"
    AML_COMPUTE_TARGET_NAME = "edocpu-compute"
    PIPELINE_VERSION = '9.9.9.9'  # NOSONAR
    CONDA_ENV_VERSION = '9.9.9.9'  # NOSONAR
    PREFIXE_INGESTION = 'IN'
    PREFIXE_TRAIN_PRED = 'TP'
    PREFIXE_NON_REGRESSION = 'NR'
    ML_PREDICTION_TYPE_DATASET_STAGING_TEST = "StagingTest"
    ML_PREDICTION_TYPE_DATASET_STAGING_PREDICTION = "StagingPrediction"
    ML_PREDICTION_TYPE_DATASET_TEST = "Test"
    ML_PREDICTION_TYPE_DATASET_PREDICTION = "Prediction"

    PIPELINE_ARG_INCOMING_FOLDER = "pipeline_arg_incoming_folder"
    PIPELINE_ARG_INPUT_FOLDER = "pipeline_arg_input_folder"
    PIPELINE_ARG_NO_ORGANISME = "no_organisme"
    PIPELINE_ARG_PREDICTION_POUR_ANNEE = "prediction_pour_annee"
    PIPELINE_ARG_PREDICTION_NUMERO = "prediction_numero"
    PIPELINE_ARG_PREDICTION_POUR_IMPUTATION_COVID = "prediction_pour_imputation_covid"
    PIPELINE_TP_STEP_REMOVE_IMPUTE = "Éliminer doubleurs / Imputer COVID"
    PIPELINE_TP_STEP_SEGREGATE = "Ségréguer"
    PIPELINE_TP_STEP_PREPARE_DATA = "Préparaton pour Machine Learning"
    PIPELINE_TP_STEP_HYPER_TUNE_TRAIN = "Rech hyperparamètres / Entrainement"
    PIPELINE_TP_STEP_PREDICT = "Prédiction"

    PIPELINE_IN_STEP_INGEST = "Ingestion données pour Machine Learning"
    PIPELINE_IN_STEP_GEN_DATASET = "Génération jeux de données"

    PIPELINE_INPUT_DATASET_NAME = "pipeline_input_dataset_name"

    PIPELINE_STATUS_COMPLETED = 'Complete'
    PIPELINE_STATUS_ERROR = 'Erreur'
    PIPELINE_STATUS_ANNULE = 'Annulé'

    INVOKE_METHOD_REST = 'rest'
    INVOKE_METHOD_SUBMIT = 'submit'

    ML_SUJET_ELEVE_RISQUE_ECHEC_4_PRIMAIRE = "ERE4P"
    ML_SUJET_ELEVE_RISQUE_ECHEC_4_PRIMAIRE_INT = 1

    ML_MODEL_SUBSET = "model_subset"
    ML_MODEL = "model"
    ML_PREDICTION_SUBSET = "prediction_subset"
    ML_PREDICTION = "prediction"

    ML_SUJET_ELEVE_RISQUE_ECHEC_PRIMAIRE = "EREP"

    # --------------------------------------
    # Nom des métriques IMPORTANT Ne pas renommer pour assurer
    # le bon fonctionnement du rapport de validation avec les expériences passées.
    # --------------------------------------
    METRIC_ACCURACY = "ACCURACY"
    METRIC_AUC_PRECISION_RECALL = "auc_precision_recall"
    METRIC_AVERAGE_PRECISION = "average_precision"
    METRIC_B_ACCURACY = "bACCURACY"
    METRIC_B_ACCURACY_CV = "bACCURACY_CV"
    METRIC_B_ACCURACY_CV_PRED = "bACCURACY_CV_predict"
    METRIC_BASE_WEIGHT = "base_weight"
    METRIC_BETA_FINAL = "beta_final"
    METRIC_C = "C"
    METRIC_CONFUSION_MATRIX = "confusion_matrix"
    METRIC_CLASS_WEIGHT = "class_weight"
    METRIC_F1_WEIGHT = "F1_weighted"
    METRIC_F1_WEIGHT_PRED = "F1_weighted_predict"
    METRIC_HYPER_PARAM = "hyper_param"
    METRIC_INTERCEPT = "intercept"
    METRIC_MATTHEW_CC = "matthewCC"
    METRIC_MATTHEW_CC_PRED = "matthewCC_predict"
    METRIC_NB_ECHEC_REEL_ANNEE_VALIDATION = "Nombre échecs réel année validation"
    METRIC_NB_ECHEC_REEL_ANNEE_PRED = "Nombre élèves réellement en échec année de prédiction"
    METRIC_PERFORMANCE = "performance"
    METRIC_PENALTY = "penalty"  # suivi
    METRIC_REPARTITION_CATEG_RISQUE_ECHEC = "repartition_categ_risque"
    METRIC_SCORE_OF_BEST_MODEL = "score_best_model"
    METRIC_SCORE1 = "score1"
    METRIC_SCORE2 = "score2"
    METRIC_SCORE1_PRED = "score1_predict"
    METRIC_SCORE2_PRED = "score2_predict"
    METRIC_SCORE_PONDERE = "score_pondere"
    METRIC_SCORE_PONDERE_PRED = "score_pondere_predict"
    METRIC_SOLVER = "solver"
    METRIC_TAUX_ECHEC_GLOBAL = "TauxEchecGlobal"
    METRIC_TAUX_PRED_EN_ECHEC = "Taux d'élèves prédit en échec"
    METRIC_TAUX_ECHEC_REEL_ANNEE_VALIDATION = "Taux échec réel année validation"
    METRIC_TAUX_ECHEC_REEL_ANNEE_PRED = "Taux élèves réellement en échec année de prédiction"
    METRIC_WEIGHT = "weight"
    METRIC_RVP_ZONE_1 = "RVP zone 1"
    METRIC_RVP_ZONE_2 = "RVP zone 2"
    METRIC_RVP_ZONE_3 = "RVP zone 3"
    METRIC_RVP_ZONE_1_PRED = "RVP zone 1 pred"
    METRIC_RVP_ZONE_2_PRED = "RVP zone 2 pred"
    METRIC_RVP_ZONE_3_PRED = "RVP zone 3 pred"
    METRIC_FAIRNESS = "mesures de fairness"
    METRIC_FEAT_AMPLITUDE_MAX = "feature avec amplitude maximale"
    METRIC_FEAT_ECRIRE_NEGATIFS = "features ecrire tous negatifs"
    METRIC_MSG_ERR = "message_err"
    METRIC_NOMAD_SOLUTION = "nomad_solution"
    METRIC_BEST_CLASSIFICATION_THRESHOLD = "best classification threshold"

    VAL_REPORT_IMAGE_IN_MOYENNE = "Moyenne"
    VAL_REPORT_IMAGE_IN_NB_VALEURS = "Nombre de valeurs"
    VAL_REPORT_IMAGE_TP_FEATURE_IMP = "Importance des features"
    VAL_REPORT_IMAGE_TP_REPART_RISQUE_ECHEC = "Répartition selon les cat"
    VAL_REPORT_IMAGE_TP_PRED_PAR_ZONE = "Prédiction par zone"
    VAL_REPORT_IMAGE_TP_MATRICE_CORRELATION = "Matrice de correlation"
    VAL_REPORT_IMAGE_TP_MATRICE_CONFUSION = "Matrice de confusion"
    VAL_REPORT_IMAGE_TP_METRIQUE_VS_THRESHOLD = "Metriques vs Threshold"
    VAL_REPORT_IMAGE_IN_BOXPLOT = "Boxplot"
    VAL_REPORT_IMAGE_IN_P9 = "p9"

    # ------------------------------------------------------------------------------------------------
    #         Seuils pour validation ML
    # Selon le moment de la prédiction, le seuil à respecter lors de la validation du ML va varier.
    # On sera moins exigeant au début de l'année scolaire
    #
    # Pour score1 on doit être en bas du seuil
    # Exemple    Si score1 > 0.40 -> erreur
    #
    # Pour MCC on doit être au dessus du seuil
    # Exemple    Si MCC < 0.29 -> erreur
    # ------------------------------------------------------------------------------------------------
    SEUIL_SCORE1_ETAPE_0 = 0.40
    SEUIL_SCORE1_ETAPE_1 = 0.35
    SEUIL_SCORE1_ETAPE_2 = 0.35

    SEUIL_MCC_ETAPE_0 = 0.15
    SEUIL_MCC_ETAPE_1 = 0.15
    SEUIL_MCC_ETAPE_2 = 0.15

    ERE4P_DataTypes = {
        'IdClient': DataType.to_string(),
        'AnneePrediction': DataType.to_string(),
        'Precision': DataType.to_string(),
        'Organisme': DataType.to_string(),
        'Fiche': DataType.to_string(),
        'Annee': DataType.to_string(),
        'Ecole': DataType.to_string(),
        'Sexe': DataType.to_string(),
        'GroupeRepere': DataType.to_string(),
        'Age30Septembre': DataType.to_long(),
        'IndFrancophone': DataType.to_long(),
        'IndEHDAA': DataType.to_long(),
        'IndPlanIntervention': DataType.to_long(),
        'ResultatFinal3eCommuniquer': DataType.to_float(),
        'ResultatFinal3eLire': DataType.to_float(),
        'ResultatFinal3eEcrire': DataType.to_float(),
        'ResultatFinal3eResoudre': DataType.to_float(),
        'ResultatFinal3eRaisonner': DataType.to_float(),
        'NbAbsencesMotivees3e': DataType.to_float(),
        'NbAbsencesNonMotivees3e': DataType.to_float(),
        'NbRetards3e': DataType.to_float(),
        'IndEchecEcritureExamen': DataType.to_float(),
        'RESULTAT_ECRITURE_ETAPE_1_4E_ANNEE': DataType.to_float(),
        'RESULTAT_LIRE_ETAPE_1_4E_ANNEE': DataType.to_float(),
        'RESULTAT_COMMUNIQUER_ETAPE_1_4E_ANNEE': DataType.to_float(),
        'RESULTAT_RAISONNER_ETAPE_1_4E_ANNEE': DataType.to_float(),
        'RESULTAT_RESOUDRE_ETAPE_1_4E_ANNEE': DataType.to_float(),
        'RESULTAT_ECRITURE_ETAPE_2_4E_ANNEE': DataType.to_float(),
        'RESULTAT_LIRE_ETAPE_2_4E_ANNEE': DataType.to_float(),
        'RESULTAT_COMMUNIQUER_ETAPE_2_4E_ANNEE': DataType.to_float(),
        'RESULTAT_RAISONNER_ETAPE_2_4E_ANNEE': DataType.to_float(),
        'RESULTAT_RESOUDRE_ETAPE_2_4E_ANNEE': DataType.to_float(),
        'NotePassageExamenEcriture': DataType.to_float(),
        'ResultatExamen': DataType.to_float(),
        'TypeDataset': DataType.to_string(),
        'Prediction': DataType.to_float(),
        'PredictionProbabilite': DataType.to_float(),
        'IndEchecEcritureExamenOriginal': DataType.to_float(),
    }

    # Pour pd.read_csv
    ERE4P_dtypes = {
        'IdClient': str,
        'AnneePrediction': str,
        'Precision': str,
        'Organisme': str,
        'Fiche': str,
        'Annee': str,
        'Ecole': str,
        'Sexe': str,
        'GroupeRepere': str,
        'Age30Septembre': int,
        'IndFrancophone': int,
        'IndEHDAA': int,
        'IndPlanIntervention': int,
        'ResultatFinal3eCommuniquer': float,
        'ResultatFinal3eLire': float,
        'ResultatFinal3eEcrire': float,
        'ResultatFinal3eResoudre': float,
        'ResultatFinal3eRaisonner': float,
        'NbAbsencesMotivees3e': float,
        'NbAbsencesNonMotivees3e': float,
        'NbRetards3e': float,
        'IndEchecEcritureExamen': float,
        'RESULTAT_ECRITURE_ETAPE_1_4E_ANNEE': float,
        'RESULTAT_LIRE_ETAPE_1_4E_ANNEE': float,
        'RESULTAT_COMMUNIQUER_ETAPE_1_4E_ANNEE': float,
        'RESULTAT_RAISONNER_ETAPE_1_4E_ANNEE': float,
        'RESULTAT_RESOUDRE_ETAPE_1_4E_ANNEE': float,
        'RESULTAT_ECRITURE_ETAPE_2_4E_ANNEE': float,
        'RESULTAT_LIRE_ETAPE_2_4E_ANNEE': float,
        'RESULTAT_COMMUNIQUER_ETAPE_2_4E_ANNEE': float,
        'RESULTAT_RAISONNER_ETAPE_2_4E_ANNEE': float,
        'RESULTAT_RESOUDRE_ETAPE_2_4E_ANNEE': float,
        'NotePassageExamenEcriture': float,
        'ResultatExamen': float,
        'TypeDataset': str,
        'Prediction': float,
        'PredictionProbabilite': float,
        'IndEchecEcritureExamenOriginal': float,
    }

    ERE4P_columns = [
        'IdClient',
        'AnneePrediction',
        'Precision',
        'Organisme',
        'Fiche',
        'Annee',
        'Ecole',
        'Sexe',
        'GroupeRepere',
        'Age30Septembre',
        'IndFrancophone',
        'IndEHDAA',
        'IndPlanIntervention',
        'ResultatFinal3eCommuniquer',
        'ResultatFinal3eLire',
        'ResultatFinal3eEcrire',
        'ResultatFinal3eResoudre',
        'ResultatFinal3eRaisonner',
        'NbAbsencesMotivees3e',
        'NbAbsencesNonMotivees3e',
        'NbRetards3e',
        'IndEchecEcritureExamen',
        'RESULTAT_ECRITURE_ETAPE_1_4E_ANNEE',
        'RESULTAT_LIRE_ETAPE_1_4E_ANNEE',
        'RESULTAT_COMMUNIQUER_ETAPE_1_4E_ANNEE',
        'RESULTAT_RAISONNER_ETAPE_1_4E_ANNEE',
        'RESULTAT_RESOUDRE_ETAPE_1_4E_ANNEE',
        'RESULTAT_ECRITURE_ETAPE_2_4E_ANNEE',
        'RESULTAT_LIRE_ETAPE_2_4E_ANNEE',
        'RESULTAT_COMMUNIQUER_ETAPE_2_4E_ANNEE',
        'RESULTAT_RAISONNER_ETAPE_2_4E_ANNEE',
        'RESULTAT_RESOUDRE_ETAPE_2_4E_ANNEE',
        'NotePassageExamenEcriture',
        'ResultatExamen',
        'TypeDataset',
        'Prediction',
        'PredictionProbabilite',
    ]

    ERE4P_columns_fairness = {
        "Sexe": "M",
        'IndFrancophone': 1,
        'IndEHDAA': 1
    }

    # TODO: On retire temporairement en sprint 32 le StatutGeneration et IMSERangDecile.
    #       Raison: Lorsque il y a des null, la fonction MetricFrame dans evaluer_fairness plante
    #       On doit donc trouver une meilleur solution en sprint 33.
    ERE6P_columns_fairness = {
        "Sexe": "M",
        # "StatutGeneration": "Non immigrant",
        'IndFrancophone': 1,
        # "IMSERangDecile": 1.0,
        "IndEHDAA": 1
    }

    # TODO: MANON TEMPORAIREMENT POUR SUPPORTER LE FAIT QUE LA TABLE DES RESULTATS ML N'EST PAS REFACTORISE SELON NOUVELLE NOMENCLATURE.
    # TODO: Manon Eliminer cet artifice.
    ERE4P_columns_AVANT_REFACTORING = [
        'IdClient',
        'AnneePrediction',
        'Precision',
        'ORG',
        'FICHE',
        'ANNEE',
        'COD_ECOLE',
        'COD_SEXE_ELEVE',
        'COD_GROUPE_REPERE',
        'AGE_30_SEPTEMBRE',
        'IND_FRANCOPHONE',
        'IND_AVEC_EHDAA',
        'IND_PLAN_INTERVENTION_EHDAA',
        'ResultatFinal3eCommuniquer',
        'ResultatFinal3eLire',
        'ResultatFinal3eEcrire',
        'ResultatFinal3eResoudre',
        'ResultatFinal3eRaisonner',
        'NbAbsencesMotivees3e',
        'NbAbsencesNonMotivees3e',
        'NbRetards3e',
        'IND_ECHEC_ECRITURE_EX_4E_ANNEE',
        'RESULTAT_ECRITURE_ETAPE_1_4E_ANNEE',
        'RESULTAT_LIRE_ETAPE_1_4E_ANNEE',
        'RESULTAT_COMMUNIQUER_ETAPE_1_4E_ANNEE',
        'RESULTAT_RAISONNER_ETAPE_1_4E_ANNEE',
        'RESULTAT_RESOUDRE_ETAPE_1_4E_ANNEE',
        'RESULTAT_ECRITURE_ETAPE_2_4E_ANNEE',
        'RESULTAT_LIRE_ETAPE_2_4E_ANNEE',
        'RESULTAT_COMMUNIQUER_ETAPE_2_4E_ANNEE',
        'RESULTAT_RAISONNER_ETAPE_2_4E_ANNEE',
        'RESULTAT_RESOUDRE_ETAPE_2_4E_ANNEE',
        'NotePassageExamenEcriture4e',
        'ResultatExamen',
        'TypeDataset',
        'Prediction',
        'PredictionProbabilite',
    ]

    ERE4P_dtypes_label = {
        'IndEchecEcritureExamen': float
    }

    ERE4P_columns_row_key = [
        'IdClient',
        'AnneePrediction',
        'Precision',
        'Organisme',
        'Fiche',
        'Ecole',
        'GroupeRepere',
        'Annee',
        'Sexe'
    ]

    ERE4P_columns_prediction = [
        'Prediction',
        'PredictionProbabilite',
    ]

    ERE4P_config_client_columns = [
        'IdClient',
        'Annee',
        'CutOffPrediction',
        'ScoreFNR',
        'ScoreTPE',
        'DatePrevu1erePrediction',
        'DatePrevu2ePrediction',
        'DatePrevu3ePrediction'
    ]

    ERE4P_config_client_DataTypes = {
        'IdClient': DataType.to_string(),
        'Annee': DataType.to_string(),
        'CutOffPrediction': DataType.to_long(),
        'ScoreFNR': DataType.to_long(),
        'ScoreTPE': DataType.to_long(),
        'DatePrevu1erePrediction': DataType.to_datetime(),
        'DatePrevu2ePrediction': DataType.to_datetime(),
        'DatePrevu3ePrediction': DataType.to_datetime()
    }

    DATE_FORMAT_STR = "%Y-%m-%d"
    DATETIME_SHORT_FORMAT_STR = "%Y-%m-%d %H:%M"

    ERE4P_mlops_columns = [
        'no_organisme',
        'prediction_pour_annee',
        'prediction_numero',
        'IN_nom_experience',
        'IN_est_fait',
        'IN_code_sortie',
        'IN_est_verifie',
        'IN_id_serie',
        'IN_date_execution',
        'TP_nom_experience',
        'TP_est_fait',
        'TP_code_sortie',
        'TP_est_verifie',
        'TP_id_serie',
        'TP_date_execution',
        'TP_est_verifie_par_assistant',
    ]

    ERE4P_mlops_dtypes = {
        'no_organisme': str,
        'prediction_pour_annee': str,
        'prediction_numero': str,
        'IN_nom_experience': str,
        'IN_est_fait': str,
        'IN_code_sortie': str,
        'IN_est_verifie': str,
        'IN_id_serie': str,
        'IN_date_execution': str,
        'TP_nom_experience': str,
        'TP_est_fait': str,
        'TP_code_sortie': str,
        'TP_est_verifie': str,
        'TP_id_serie': str,
        'TP_date_execution': str,
        'TP_est_verifie_par_assistant': str,
    }

    # Colonne pour la verification des zones d'echec pour chaque moment
    ERE4P_colonne_zone_echec = {
        '0': 'ResultatFinal3eEcrire',
        '1': 'RESULTAT_ECRITURE_ETAPE_1_4E_ANNEE',
        '2': 'RESULTAT_ECRITURE_ETAPE_2_4E_ANNEE'
    }

    ERE6P_mlops_columns = ERE4P_mlops_columns
    ERE6P_mlops_dtypes = ERE4P_mlops_dtypes

    # ---------------------------------------------------------------
    # Section ERE6P
    # ---------------------------------------------------------------
    ML_SUJET_ELEVE_RISQUE_ECHEC_6_PRIMAIRE = "ERE6P"
    ML_SUJET_ELEVE_RISQUE_ECHEC_6_PRIMAIRE_INT = 2

    ERE6P_columns_core = [
        'IdClient',
        'AnneePrediction',
        'Precision',
        'Organisme',
        'Fiche',
        'Annee',
        'Sexe',
        'GroupeRepere',
        'Age30Septembre',
        'IndFrancophone',
        'IndDoubleurAnneePrecedente',
        'IndDoubleurAnneeCourante',
        'NbAnneesRedoublees',
        'IndRedoublement',
        'RepondantPere',
        'RepondantMere',
        'RepondantTuteur',
        'InterdictionPere',
        'InterdictionMere',
        'InterdictionTuteur',
        'DecesPere',
        'DecesMere',
        # 'OccupationPere',
        # 'OccupationMere',
        'ScolaritePere',
        'ScolariteMere',
        'LieuNaissance',
        'CategorieLinguistique',
        'StatutGeneration',
    ]

    # Info de 6e
    ERE6P_columns_niv_6 = [
        'Ecole',
        'ResEtape1Ecrire',
        'ResEtape2Ecrire',
        'ResEtape1Lire',
        'ResEtape2Lire',
        'ResEtape1Communiquer',
        'ResEtape2Communiquer',
        'ResEtape1Raisonner',
        'ResEtape2Raisonner',
        'ResEtape1Resoudre',
        'ResEtape2Resoudre',
        'ResEtape1CommuniquerLangueSeconde',
        'ResEtape2CommuniquerLangueSeconde',
        'ResEtape1LireLangueSeconde',
        'ResEtape2LireLangueSeconde',
        'ResEtape1EcrireLangueSeconde',
        'ResEtape2EcrireLangueSeconde',
        'NbAbsencesMotiveesEtape1',
        'NbAbsencesNonMotiveesEtape1',
        'NbRetardsEtape1',
        'NbAbsencesMotiveesEtape2',
        'NbAbsencesNonMotiveesEtape2',
        'NbRetardsEtape2',
        'IndPlanIntervention',
        'IndEHDAA',
        'IndMesureAide',
        'IMSERangDecile',
        'NbEcolesFrequentees',
    ]

    # Historique 3e année
    ERE6P_columns_hist_niv_3 = [
        # 'Ecole3e',
        'ResFinal3eCommuniquer',
        'ResFinal3eLire',
        'ResFinal3eEcrire',
        'ResFinal3eResoudre',
        'ResFinal3eRaisonner',
        'ResFinal3eCommuniquerLangueSeconde',
        'ResFinal3eLireLangueSeconde',
        'ResFinal3eEcrireLangueSeconde',
        'NbAbsencesMotivees3eEtape1',
        'NbAbsencesNonMotivees3eEtape1',
        'NbRetards3eEtape1',
        'NbAbsencesMotivees3eEtape2',
        'NbAbsencesNonMotivees3eEtape2',
        'NbRetards3eEtape2',
        'NbAbsencesMotivees3eEtape3',
        'NbAbsencesNonMotivees3eEtape3',
        'NbRetards3eEtape3',
        'IndPlanIntervention3e',
        'IndEHDAA3e',
        'IndMesureAide3e',
    ]

    # Historique 4e année
    ERE6P_columns_hist_niv_4 = [
        # 'Ecole4e',
        'ResFinal4eCommuniquer',
        'ResFinal4eLire',
        'ResFinal4eEcrire',
        'ResFinal4eResoudre',
        'ResFinal4eRaisonner',
        'ResFinal4eCommuniquerLangueSeconde',
        'ResFinal4eLireLangueSeconde',
        'ResFinal4eEcrireLangueSeconde',
        'Res4eEtape1Ecrire',
        'Res4eEtape2Ecrire',
        'Res4eEtape3Ecrire',
        'Res4eEtape1Lire',
        'Res4eEtape2Lire',
        'Res4eEtape3Lire',
        'Res4eEtape1Communiquer',
        'Res4eEtape2Communiquer',
        'Res4eEtape3Communiquer',
        'Res4eEtape1Raisonner',
        'Res4eEtape2Raisonner',
        'Res4eEtape3Raisonner',
        'Res4eEtape1Resoudre',
        'Res4eEtape2Resoudre',
        'Res4eEtape3Resoudre',
        'Res4eEtape1CommuniquerLangueSeconde',
        'Res4eEtape2CommuniquerLangueSeconde',
        'Res4eEtape3CommuniquerLangueSeconde',
        'Res4eEtape1LireLangueSeconde',
        'Res4eEtape2LireLangueSeconde',
        'Res4eEtape3LireLangueSeconde',
        'Res4eEtape1EcrireLangueSeconde',
        'Res4eEtape2EcrireLangueSeconde',
        'Res4eEtape3EcrireLangueSeconde',
        'NbAbsencesMotivees4eEtape1',
        'NbAbsencesNonMotivees4eEtape1',
        'NbRetards4eEtape1',
        'NbAbsencesMotivees4eEtape2',
        'NbAbsencesNonMotivees4eEtape2',
        'NbRetards4eEtape2',
        'NbAbsencesMotivees4eEtape3',
        'NbAbsencesNonMotivees4eEtape3',
        'NbRetards4eEtape3',
        'IndPlanIntervention4e',
        'IndEHDAA4e',
        'IndMesureAide4e',
        'ResExamen4e',
    ]

    # Historique 5e année
    ERE6P_columns_hist_niv_5 = [
        # 'Ecole5e',
        'ResFinal5eCommuniquer',
        'ResFinal5eLire',
        'ResFinal5eEcrire',
        'ResFinal5eResoudre',
        'ResFinal5eRaisonner',
        'ResFinal5eCommuniquerLangueSeconde',
        'ResFinal5eLireLangueSeconde',
        'ResFinal5eEcrireLangueSeconde',
        'Res5eEtape1Ecrire',
        'Res5eEtape2Ecrire',
        'Res5eEtape3Ecrire',
        'Res5eEtape1Lire',
        'Res5eEtape2Lire',
        'Res5eEtape3Lire',
        'Res5eEtape1Communiquer',
        'Res5eEtape2Communiquer',
        'Res5eEtape3Communiquer',
        'Res5eEtape1Raisonner',
        'Res5eEtape2Raisonner',
        'Res5eEtape3Raisonner',
        'Res5eEtape1Resoudre',
        'Res5eEtape2Resoudre',
        'Res5eEtape3Resoudre',
        'Res5eEtape1CommuniquerLangueSeconde',
        'Res5eEtape2CommuniquerLangueSeconde',
        'Res5eEtape3CommuniquerLangueSeconde',
        'Res5eEtape1LireLangueSeconde',
        'Res5eEtape2LireLangueSeconde',
        'Res5eEtape3LireLangueSeconde',
        'Res5eEtape1EcrireLangueSeconde',
        'Res5eEtape2EcrireLangueSeconde',
        'Res5eEtape3EcrireLangueSeconde',
        'NbAbsencesMotivees5eEtape1',
        'NbAbsencesNonMotivees5eEtape1',
        'NbRetards5eEtape1',
        'NbAbsencesMotivees5eEtape2',
        'NbAbsencesNonMotivees5eEtape2',
        'NbRetards5eEtape2',
        'NbAbsencesMotivees5eEtape3',
        'NbAbsencesNonMotivees5eEtape3',
        'NbRetards5eEtape3',
        'IndPlanIntervention5e',
        'IndEHDAA5e',
        'IndMesureAide5e',
        'IMSERangDecile5e',
        'NbEcolesFrequentees5e',
    ]

    # Info 6e détaillant le contexte du label
    ERE6P_columns_niv_6_label = [
        'NotePassageExamenEcriture',
        'ResExamen',
        'IndEchecEcritureExamen',       # Notre label
    ]

    # Info prediction
    ERE6P_columns_niv_6_prediction = [
        'ModelePredictif',
        'TypeDataset',                  # Staging ou non
        'Prediction',
        'PredictionProbabilite',
    ]

    # Définition des colonnes de ERE6P

    ERE6P_columns = ERE6P_columns_core + \
        ERE6P_columns_niv_6 + \
        ERE6P_columns_hist_niv_5 + \
        ERE6P_columns_hist_niv_4 + \
        ERE6P_columns_hist_niv_3 + \
        ERE6P_columns_niv_6_label + \
        ERE6P_columns_niv_6_prediction

    ERE6P_dtypes_label = {
        'IndEchecEcritureExamen': float
    }

    ERE6P_columns_prediction = [
        'Prediction',
        'PredictionProbabilite',
    ]

    ERE6P_DataTypes = {
        'IdClient': DataType.to_string(),
        'AnneePrediction': DataType.to_string(),
        'Precision': DataType.to_string(),
        'Organisme': DataType.to_string(),
        'Fiche': DataType.to_string(),
        'Annee': DataType.to_string(),
        'Sexe': DataType.to_string(),
        'GroupeRepere': DataType.to_string(),
        'Age30Septembre': DataType.to_long(),
        'IndFrancophone': DataType.to_long(),
        'IndDoubleurAnneePrecedente': DataType.to_long(),
        'IndDoubleurAnneeCourante': DataType.to_long(),
        'NbAnneesRedoublees': DataType.to_long(),
        'IndRedoublement': DataType.to_long(),
        'RepondantPere': DataType.to_long(),
        'RepondantMere': DataType.to_long(),
        'RepondantTuteur': DataType.to_long(),
        'InterdictionPere': DataType.to_long(),
        'InterdictionMere': DataType.to_long(),
        'InterdictionTuteur': DataType.to_long(),
        'DecesPere': DataType.to_long(),
        'DecesMere': DataType.to_long(),
        # 'OccupationPere': DataType.to_string(),
        # 'OccupationMere': DataType.to_string(),
        'ScolaritePere': DataType.to_string(),
        'ScolariteMere': DataType.to_string(),
        'LieuNaissance': DataType.to_string(),
        'CategorieLinguistique': DataType.to_string(),
        'StatutGeneration': DataType.to_string(),

        # Info 6e
        'Ecole': DataType.to_string(),
        'ResEtape1Ecrire': DataType.to_float(),
        'ResEtape2Ecrire': DataType.to_float(),
        'ResEtape1Lire': DataType.to_float(),
        'ResEtape2Lire': DataType.to_float(),
        'ResEtape1Communiquer': DataType.to_float(),
        'ResEtape2Communiquer': DataType.to_float(),
        'ResEtape1Raisonner': DataType.to_float(),
        'ResEtape2Raisonner': DataType.to_float(),
        'ResEtape1Resoudre': DataType.to_float(),
        'ResEtape2Resoudre': DataType.to_float(),
        'ResEtape1CommuniquerLangueSeconde': DataType.to_float(),
        'ResEtape2CommuniquerLangueSeconde': DataType.to_float(),
        'ResEtape1LireLangueSeconde': DataType.to_float(),
        'ResEtape2LireLangueSeconde': DataType.to_float(),
        'ResEtape1EcrireLangueSeconde': DataType.to_float(),
        'ResEtape2EcrireLangueSeconde': DataType.to_float(),
        'NbAbsencesMotiveesEtape1': DataType.to_float(),
        'NbAbsencesNonMotiveesEtape1': DataType.to_float(),
        'NbRetardsEtape1': DataType.to_float(),
        'NbAbsencesMotiveesEtape2': DataType.to_float(),
        'NbAbsencesNonMotiveesEtape2': DataType.to_float(),
        'NbRetardsEtape2': DataType.to_float(),
        'IndPlanIntervention': DataType.to_long(),
        'IndEHDAA': DataType.to_long(),
        'IndMesureAide': DataType.to_long(),
        'IMSERangDecile': DataType.to_float(),
        'NbEcolesFrequentees': DataType.to_float(),

        # Historique 3e année
        # 'Ecole3e': DataType.to_string(),
        'ResFinal3eCommuniquer': DataType.to_float(),
        'ResFinal3eLire': DataType.to_float(),
        'ResFinal3eEcrire': DataType.to_float(),
        'ResFinal3eResoudre': DataType.to_float(),
        'ResFinal3eRaisonner': DataType.to_float(),
        'ResFinal3eCommuniquerLangueSeconde': DataType.to_float(),
        'ResFinal3eLireLangueSeconde': DataType.to_float(),
        'ResFinal3eEcrireLangueSeconde': DataType.to_float(),
        'NbAbsencesMotivees3eEtape1': DataType.to_float(),
        'NbAbsencesNonMotivees3eEtape1': DataType.to_float(),
        'NbRetards3eEtape1': DataType.to_float(),
        'NbAbsencesMotivees3eEtape2': DataType.to_float(),
        'NbAbsencesNonMotivees3eEtape2': DataType.to_float(),
        'NbRetards3eEtape2': DataType.to_float(),
        'NbAbsencesMotivees3eEtape3': DataType.to_float(),
        'NbAbsencesNonMotivees3eEtape3': DataType.to_float(),
        'NbRetards3eEtape3': DataType.to_float(),
        'IndPlanIntervention3e': DataType.to_long(),
        'IndEHDAA3e': DataType.to_long(),
        'IndMesureAide3e': DataType.to_long(),

        # Historique 4e année
        # 'Ecole4e': DataType.to_string(),
        'ResFinal4eCommuniquer': DataType.to_float(),
        'ResFinal4eLire': DataType.to_float(),
        'ResFinal4eEcrire': DataType.to_float(),
        'ResFinal4eResoudre': DataType.to_float(),
        'ResFinal4eRaisonner': DataType.to_float(),
        'ResFinal4eCommuniquerLangueSeconde': DataType.to_float(),
        'ResFinal4eLireLangueSeconde': DataType.to_float(),
        'ResFinal4eEcrireLangueSeconde': DataType.to_float(),
        'Res4eEtape1Ecrire': DataType.to_float(),
        'Res4eEtape2Ecrire': DataType.to_float(),
        'Res4eEtape3Ecrire': DataType.to_float(),
        'Res4eEtape1Lire': DataType.to_float(),
        'Res4eEtape2Lire': DataType.to_float(),
        'Res4eEtape3Lire': DataType.to_float(),
        'Res4eEtape1Communiquer': DataType.to_float(),
        'Res4eEtape2Communiquer': DataType.to_float(),
        'Res4eEtape3Communiquer': DataType.to_float(),
        'Res4eEtape1Raisonner': DataType.to_float(),
        'Res4eEtape2Raisonner': DataType.to_float(),
        'Res4eEtape3Raisonner': DataType.to_float(),
        'Res4eEtape1Resoudre': DataType.to_float(),
        'Res4eEtape2Resoudre': DataType.to_float(),
        'Res4eEtape3Resoudre': DataType.to_float(),
        'Res4eEtape1CommuniquerLangueSeconde': DataType.to_float(),
        'Res4eEtape2CommuniquerLangueSeconde': DataType.to_float(),
        'Res4eEtape3CommuniquerLangueSeconde': DataType.to_float(),
        'Res4eEtape1LireLangueSeconde': DataType.to_float(),
        'Res4eEtape2LireLangueSeconde': DataType.to_float(),
        'Res4eEtape3LireLangueSeconde': DataType.to_float(),
        'Res4eEtape1EcrireLangueSeconde': DataType.to_float(),
        'Res4eEtape2EcrireLangueSeconde': DataType.to_float(),
        'Res4eEtape3EcrireLangueSeconde': DataType.to_float(),
        'NbAbsencesMotivees4eEtape1': DataType.to_float(),
        'NbAbsencesNonMotivees4eEtape1': DataType.to_float(),
        'NbRetards4eEtape1': DataType.to_float(),
        'NbAbsencesMotivees4eEtape2': DataType.to_float(),
        'NbAbsencesNonMotivees4eEtape2': DataType.to_float(),
        'NbRetards4eEtape2': DataType.to_float(),
        'NbAbsencesMotivees4eEtape3': DataType.to_float(),
        'NbAbsencesNonMotivees4eEtape3': DataType.to_float(),
        'NbRetards4eEtape3': DataType.to_float(),
        'IndPlanIntervention4e': DataType.to_long(),
        'IndEHDAA4e': DataType.to_long(),
        'IndMesureAide4e': DataType.to_long(),
        'ResExamen4e': DataType.to_float(),

        # Historique 5e année
        # 'Ecole5e': DataType.to_string(),
        'ResFinal5eCommuniquer': DataType.to_float(),
        'ResFinal5eLire': DataType.to_float(),
        'ResFinal5eEcrire': DataType.to_float(),
        'ResFinal5eResoudre': DataType.to_float(),
        'ResFinal5eRaisonner': DataType.to_float(),
        'ResFinal5eCommuniquerLangueSeconde': DataType.to_float(),
        'ResFinal5eLireLangueSeconde': DataType.to_float(),
        'ResFinal5eEcrireLangueSeconde': DataType.to_float(),
        'Res5eEtape1Ecrire': DataType.to_float(),
        'Res5eEtape2Ecrire': DataType.to_float(),
        'Res5eEtape3Ecrire': DataType.to_float(),
        'Res5eEtape1Lire': DataType.to_float(),
        'Res5eEtape2Lire': DataType.to_float(),
        'Res5eEtape3Lire': DataType.to_float(),
        'Res5eEtape1Communiquer': DataType.to_float(),
        'Res5eEtape2Communiquer': DataType.to_float(),
        'Res5eEtape3Communiquer': DataType.to_float(),
        'Res5eEtape1Raisonner': DataType.to_float(),
        'Res5eEtape2Raisonner': DataType.to_float(),
        'Res5eEtape3Raisonner': DataType.to_float(),
        'Res5eEtape1Resoudre': DataType.to_float(),
        'Res5eEtape2Resoudre': DataType.to_float(),
        'Res5eEtape3Resoudre': DataType.to_float(),
        'Res5eEtape1CommuniquerLangueSeconde': DataType.to_float(),
        'Res5eEtape2CommuniquerLangueSeconde': DataType.to_float(),
        'Res5eEtape3CommuniquerLangueSeconde': DataType.to_float(),
        'Res5eEtape1LireLangueSeconde': DataType.to_float(),
        'Res5eEtape2LireLangueSeconde': DataType.to_float(),
        'Res5eEtape3LireLangueSeconde': DataType.to_float(),
        'Res5eEtape1EcrireLangueSeconde': DataType.to_float(),
        'Res5eEtape2EcrireLangueSeconde': DataType.to_float(),
        'Res5eEtape3EcrireLangueSeconde': DataType.to_float(),
        'NbAbsencesMotivees5eEtape1': DataType.to_float(),
        'NbAbsencesNonMotivees5eEtape1': DataType.to_float(),
        'NbRetards5eEtape1': DataType.to_float(),
        'NbAbsencesMotivees5eEtape2': DataType.to_float(),
        'NbAbsencesNonMotivees5eEtape2': DataType.to_float(),
        'NbRetards5eEtape2': DataType.to_float(),
        'NbAbsencesMotivees5eEtape3': DataType.to_float(),
        'NbAbsencesNonMotivees5eEtape3': DataType.to_float(),
        'NbRetards5eEtape3': DataType.to_float(),
        'IndPlanIntervention5e': DataType.to_long(),
        'IndEHDAA5e': DataType.to_long(),
        'IndMesureAide5e': DataType.to_long(),
        'IMSERangDecile5e': DataType.to_float(),
        'NbEcolesFrequentees5e': DataType.to_float(),

        # Info 6e
        'NotePassageExamenEcriture': DataType.to_float(),
        'ResExamen': DataType.to_float(),
        'IndEchecEcritureExamen': DataType.to_float(),

        # Info prediction
        'ModelePredictif': DataType.to_long(),
        'TypeDataset': DataType.to_string(),
        'Prediction': DataType.to_float(),
        'PredictionProbabilite': DataType.to_float(),
        'IndEchecEcritureExamenOriginal': DataType.to_float(),
    }

    # Pour pd.read_csv
    ERE6P_dtypes = {
        'IdClient': str,
        'AnneePrediction': str,
        'Precision': str,
        'Organisme': str,
        'Fiche': str,
        'Annee': str,
        'Sexe': str,
        'GroupeRepere': str,
        'Age30Septembre': int,
        'IndFrancophone': int,
        'IndEHDAA': int,
        'IndPlanIntervention': int,
        'IndDoubleurAnneePrecedente': int,
        'IndDoubleurAnneeCourante': int,
        'NbAnneesRedoublees': int,
        'IndRedoublement': int,
        'RepondantPere': int,
        'RepondantMere': int,
        'RepondantTuteur': int,
        'InterdictionPere': int,
        'InterdictionMere': int,
        'InterdictionTuteur': int,
        'DecesPere': int,
        'DecesMere': int,
        # 'OccupationPere': str,
        # 'OccupationMere': str,
        'ScolaritePere': str,
        'ScolariteMere': str,
        'LieuNaissance': str,
        'CategorieLinguistique': str,
        'StatutGeneration': str,

        # Info 6e
        'Ecole': str,
        'ResEtape1Ecrire': float,
        'ResEtape2Ecrire': float,
        'ResEtape1Lire': float,
        'ResEtape2Lire': float,
        'ResEtape1Communiquer': float,
        'ResEtape2Communiquer': float,
        'ResEtape1Raisonner': float,
        'ResEtape2Raisonner': float,
        'ResEtape1Resoudre': float,
        'ResEtape2Resoudre': float,
        'ResEtape1CommuniquerLangueSeconde': float,
        'ResEtape2CommuniquerLangueSeconde': float,
        'ResEtape1LireLangueSeconde': float,
        'ResEtape2LireLangueSeconde': float,
        'ResEtape1EcrireLangueSeconde': float,
        'ResEtape2EcrireLangueSeconde': float,
        'NbAbsencesMotiveesEtape1': float,
        'NbAbsencesNonMotiveesEtape1': float,
        'NbRetardsEtape1': float,
        'NbAbsencesMotiveesEtape2': float,
        'NbAbsencesNonMotiveesEtape2': float,
        'NbRetardsEtape2': float,
        'IndPlanIntervention': int,
        'IndEHDAA': int,
        'IndMesureAide': int,
        'IMSERangDecile': float,
        'NbEcolesFrequentees': float,

        # Historique 3e année
        # 'Ecole3e': str,
        'ResFinal3eCommuniquer': float,
        'ResFinal3eLire': float,
        'ResFinal3eEcrire': float,
        'ResFinal3eResoudre': float,
        'ResFinal3eRaisonner': float,
        'ResFinal3eCommuniquerLangueSeconde': float,
        'ResFinal3eLireLangueSeconde': float,
        'ResFinal3eEcrireLangueSeconde': float,
        'NbAbsencesMotivees3eEtape1': float,
        'NbAbsencesNonMotivees3eEtape1': float,
        'NbRetards3eEtape1': float,
        'NbAbsencesMotivees3eEtape2': float,
        'NbAbsencesNonMotivees3eEtape2': float,
        'NbRetards3eEtape2': float,
        'NbAbsencesMotivees3eEtape3': float,
        'NbAbsencesNonMotivees3eEtape3': float,
        'NbRetards3eEtape3': float,
        'IndPlanIntervention3e': int,
        'IndEHDAA3e': int,
        'IndMesureAide3e': int,

        # Historique 4e année
        # 'Ecole4e': str,
        'ResFinal4eCommuniquer': float,
        'ResFinal4eLire': float,
        'ResFinal4eEcrire': float,
        'ResFinal4eResoudre': float,
        'ResFinal4eRaisonner': float,
        'ResFinal4eCommuniquerLangueSeconde': float,
        'ResFinal4eLireLangueSeconde': float,
        'ResFinal4eEcrireLangueSeconde': float,
        'Res4eEtape1Ecrire': float,
        'Res4eEtape2Ecrire': float,
        'Res4eEtape3Ecrire': float,
        'Res4eEtape1Lire': float,
        'Res4eEtape2Lire': float,
        'Res4eEtape3Lire': float,
        'Res4eEtape1Communiquer': float,
        'Res4eEtape2Communiquer': float,
        'Res4eEtape3Communiquer': float,
        'Res4eEtape1Raisonner': float,
        'Res4eEtape2Raisonner': float,
        'Res4eEtape3Raisonner': float,
        'Res4eEtape1Resoudre': float,
        'Res4eEtape2Resoudre': float,
        'Res4eEtape3Resoudre': float,
        'Res4eEtape1CommuniquerLangueSeconde': float,
        'Res4eEtape2CommuniquerLangueSeconde': float,
        'Res4eEtape3CommuniquerLangueSeconde': float,
        'Res4eEtape1LireLangueSeconde': float,
        'Res4eEtape2LireLangueSeconde': float,
        'Res4eEtape3LireLangueSeconde': float,
        'Res4eEtape1EcrireLangueSeconde': float,
        'Res4eEtape2EcrireLangueSeconde': float,
        'Res4eEtape3EcrireLangueSeconde': float,
        'NbAbsencesMotivees4eEtape1': float,
        'NbAbsencesNonMotivees4eEtape1': float,
        'NbRetards4eEtape1': float,
        'NbAbsencesMotivees4eEtape2': float,
        'NbAbsencesNonMotivees4eEtape2': float,
        'NbRetards4eEtape2': float,
        'NbAbsencesMotivees4eEtape3': float,
        'NbAbsencesNonMotivees4eEtape3': float,
        'NbRetards4eEtape3': float,
        'IndPlanIntervention4e': int,
        'IndEHDAA4e': int,
        'IndMesureAide4e': int,
        'ResExamen4e': float,

        # Historique 5e année
        # 'Ecole5e': str,
        'ResFinal5eCommuniquer': float,
        'ResFinal5eLire': float,
        'ResFinal5eEcrire': float,
        'ResFinal5eResoudre': float,
        'ResFinal5eRaisonner': float,
        'ResFinal5eCommuniquerLangueSeconde': float,
        'ResFinal5eLireLangueSeconde': float,
        'ResFinal5eEcrireLangueSeconde': float,
        'Res5eEtape1Ecrire': float,
        'Res5eEtape2Ecrire': float,
        'Res5eEtape3Ecrire': float,
        'Res5eEtape1Lire': float,
        'Res5eEtape2Lire': float,
        'Res5eEtape3Lire': float,
        'Res5eEtape1Communiquer': float,
        'Res5eEtape2Communiquer': float,
        'Res5eEtape3Communiquer': float,
        'Res5eEtape1Raisonner': float,
        'Res5eEtape2Raisonner': float,
        'Res5eEtape3Raisonner': float,
        'Res5eEtape1Resoudre': float,
        'Res5eEtape2Resoudre': float,
        'Res5eEtape3Resoudre': float,
        'Res5eEtape1CommuniquerLangueSeconde': float,
        'Res5eEtape2CommuniquerLangueSeconde': float,
        'Res5eEtape3CommuniquerLangueSeconde': float,
        'Res5eEtape1LireLangueSeconde': float,
        'Res5eEtape2LireLangueSeconde': float,
        'Res5eEtape3LireLangueSeconde': float,
        'Res5eEtape1EcrireLangueSeconde': float,
        'Res5eEtape2EcrireLangueSeconde': float,
        'Res5eEtape3EcrireLangueSeconde': float,
        'NbAbsencesMotivees5eEtape1': float,
        'NbAbsencesNonMotivees5eEtape1': float,
        'NbRetards5eEtape1': float,
        'NbAbsencesMotivees5eEtape2': float,
        'NbAbsencesNonMotivees5eEtape2': float,
        'NbRetards5eEtape2': float,
        'NbAbsencesMotivees5eEtape3': float,
        'NbAbsencesNonMotivees5eEtape3': float,
        'NbRetards5eEtape3': float,
        'IndPlanIntervention5e': int,
        'IndEHDAA5e': int,
        'IndMesureAide5e': int,
        'IMSERangDecile5e': float,
        'NbEcolesFrequentees5e': float,

        # Info 6e
        'NotePassageExamenEcriture': float,
        'ResExamen': float,
        'IndEchecEcritureExamen': float,

        # Info prediction
        'ModelePredictif': int,
        'TypeDataset': str,
        'Prediction': float,
        'PredictionProbabilite': float,
        'IndEchecEcritureExamenOriginal': float,

    }

    # Colonne pour la verification des zones d'echec pour chaque moment
    ERE6P_colonne_zone_echec = {
        '0': 'ResFinal5eEcrire',
        '1': 'ResEtape1Ecrire',
        '2': 'ResEtape2Ecrire'
    }

    def __setattr__(self, *_):
        pass


def parse_myargs():
    '''Lecture des paramètres pour l'exécution'''

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--autodeploy',
        type=str,
        help="Permet de copier le script dans les dossiers standards d'un pipeline.  [ingest, prepare]",
        default=0
    )

    opts = parser.parse_args()
    return opts


def get_aml_acr_name(workspace):
    """
    Retracer le azure container registry associé au Workspace Azure ML
    """
    info_dict = workspace.get_details()

    acr_key = 'containerRegistry'
    containerRegistry = info_dict[acr_key]

    print("Container registry associé au workspace = ", info_dict[acr_key])

    acr_name = containerRegistry[containerRegistry.rindex(r"/") + 1:]

    return acr_name


def get_ml_sujet_analyse():
    sujet = os.environ['ML_SUJET']
    return sujet


def get_ml_sql_datastore_name():
    datastore_name = os.environ['ML_SQL_DATASTORE_NAME']
    return datastore_name


def get_ml_methode_analyse():
    methode = os.environ['ML_METHODE']
    return methode


def get_ml_timeout_sleep_value_checker():
    timeout = int(os.environ['ML_TIMEOUT_SLEEP_VALUE_CHECKER'])
    return timeout


def get_ml_timeout_pipeline():
    timeout = int(os.environ['ML_TIMEOUT_PIPELINE'])
    return timeout


def get_ml_timeout_sql():
    timeout = int(os.environ['ML_TIMEOUT_SQL'])
    return timeout


def get_ml_timeout_sql_connexion():
    timeout = int(os.environ['ML_TIMEOUT_SQL_CONNEXION'])
    return timeout


def get_sql_connexion(workspace: Workspace):

    connectstring = get_sql_connectstring(workspace)

    max_try = 3

    for nb_try in range(max_try):

        try:

            connection = pyodbc.connect(connectstring)
            if connection:
                return connection
        except Exception:
            print("Try établir connexion vers SQL : Tentative no ", nb_try)

    raise Exception("EDOTrace Impossible d'établir la communication avec le serveur SQL")


def get_ml_timeout_clean_submit():
    timeout = int(os.environ['ML_TIMEOUT_CLEAN_SUBMIT'])
    return timeout


def get_ml_sleep_value_max_iteration():

    sleep_value_max_iteration = get_ml_timeout_pipeline() / get_ml_timeout_sleep_value_checker()
    return sleep_value_max_iteration


def get_experiment_name_non_regression(sujet: str, methode: str, fromenv: str):
    """ Retourne le nom d'un experiment dans AML."""
    experiment_name = "{}_{}_{}_{}".format(sujet, MLOPSCONST.PREFIXE_NON_REGRESSION, methode, fromenv)
    return experiment_name


def get_experiment_name(sujet: str, methode: str, fromenv: str):
    """ Retourne le nom d'un experiment dans AML."""
    experiment_name = "{}_{}_{}".format(sujet, methode, fromenv)
    return experiment_name


def get_experiment_name_ingestion(sujet: str,
                                  no_organisme: str,
                                  prediction_pour_annee: int,
                                  prediction_numero: str):
    experiment_name = "{}_{}_{}_{}_{}".format(MLOPSCONST.PREFIXE_INGESTION, sujet, no_organisme, prediction_pour_annee, prediction_numero)
    return experiment_name


def get_experiment_name_train_pred(sujet: str,
                                   no_organisme: str,
                                   prediction_pour_annee: int,
                                   prediction_numero: str):
    experiment_name = "{}_{}_{}_{}_{}".format(MLOPSCONST.PREFIXE_TRAIN_PRED, sujet, no_organisme, prediction_pour_annee, prediction_numero)
    return experiment_name


def get_pipeline_name_ingest():
    """ Retourne le nom d'un pipeline dans AML."""

    sujet = get_ml_sujet_analyse()
    methode = get_ml_methode_analyse()

    pipeline_name = "{}_pipeline_ingestion_{}".format(sujet, methode)
    return pipeline_name


def get_pipeline_name_train():
    """ Retourne le nom d'un pipeline dans AML."""

    sujet = get_ml_sujet_analyse()
    methode = get_ml_methode_analyse()

    pipeline_name = "{}_pipeline_entrainement_{}".format(sujet, methode)
    return pipeline_name


def get_pipeline_endpoint_name_train():
    """ Retourne le nom d'un endpoint officiel dans AML."""
    sujet = get_ml_sujet_analyse()
    methode = get_ml_methode_analyse()

    pipeline_name = "{}_pipeline_entrainement_{}".format(sujet, methode)
    return pipeline_name


def get_pipeline_endpoint_name_ingest():
    """ Retourne le nom d'un endpoint officiel dans AML."""
    sujet = get_ml_sujet_analyse()
    methode = get_ml_methode_analyse()

    pipeline_name = "{}_pipeline_ingestion_{}".format(sujet, methode)
    return pipeline_name


def get_conda_env_name():
    sujet = get_ml_sujet_analyse()
    methode = get_ml_methode_analyse()

    conda_env_name = "{}_{}".format(sujet, methode)
    return conda_env_name


def get_conda_env_specification(__file__):
    root_folder = get_root_folder_caller(__file__)
    subfolder = "{}_{}".format(get_ml_sujet_analyse(), get_ml_methode_analyse())
    conda_env_specification = os.path.join(root_folder, "environments", subfolder, MLOPSCONST.CONDA_SPECIFICATIONS_YML)
    return conda_env_specification


def get_conda_env_variables():
    env_variables = {
        "ML_SUJET": get_ml_sujet_analyse(),
        "ML_METHODE": get_ml_methode_analyse(),
        "ML_SQL_DATASTORE_NAME": get_ml_sql_datastore_name(),
        "ML_TIMEOUT_PIPELINE": get_ml_timeout_pipeline(),
        "ML_TIMEOUT_SQL": get_ml_timeout_sql(),
        "ML_TIMEOUT_SQL_CONNEXION": get_ml_timeout_sql_connexion(),
        "ML_TIMEOUT_SLEEP_VALUE_CHECKER": get_ml_timeout_sleep_value_checker(),
        "ML_TIMEOUT_CLEAN_SUBMIT": get_ml_timeout_clean_submit(),
    }
    return env_variables


def get_docker_specification_fullpath(__file__):
    root_folder = get_root_folder_caller(__file__)

    fullpath = os.path.join(root_folder, "environments", "Docker", MLOPSCONST.DOCKER_SPECIFICATIONS_FILE)
    return fullpath


def clean_submit(run: Run):
    # ----------------------------------------------------------------------
    # Nettoyage du dossier C:\Users\girardm\AppData\Local\Temp\azureml_runs
    # ----------------------------------------------------------------------
    print("EDOTrace Nettoyage")

    while True:
        # Délai nécessaire car j'avais une erreur de droit d'accès lors du clean
        sleep(get_ml_timeout_clean_submit())
        run.clean()
        break


def get_script_name_ingest():

    sujet = get_ml_sujet_analyse()

    if (sujet == MLOPSCONST.ML_SUJET_ELEVE_RISQUE_ECHEC_4_PRIMAIRE
            or sujet == MLOPSCONST.ML_SUJET_ELEVE_RISQUE_ECHEC_6_PRIMAIRE):
        script_name = "ingest_data_{}_{}.py".format(MLOPSCONST.ML_SUJET_ELEVE_RISQUE_ECHEC_PRIMAIRE, get_ml_methode_analyse())
    return script_name


def get_script_name_prepare():

    sujet = get_ml_sujet_analyse()

    if (sujet == MLOPSCONST.ML_SUJET_ELEVE_RISQUE_ECHEC_4_PRIMAIRE
            or sujet == MLOPSCONST.ML_SUJET_ELEVE_RISQUE_ECHEC_6_PRIMAIRE):
        script_name = "prepare_data_{}_{}.py".format(MLOPSCONST.ML_SUJET_ELEVE_RISQUE_ECHEC_PRIMAIRE, get_ml_methode_analyse())
    return script_name


def get_script_name_train():

    sujet = get_ml_sujet_analyse()

    if (sujet == MLOPSCONST.ML_SUJET_ELEVE_RISQUE_ECHEC_4_PRIMAIRE
            or sujet == MLOPSCONST.ML_SUJET_ELEVE_RISQUE_ECHEC_6_PRIMAIRE):
        script_name = "train_{}_{}.py".format(MLOPSCONST.ML_SUJET_ELEVE_RISQUE_ECHEC_PRIMAIRE, get_ml_methode_analyse())
    return script_name


def get_script_name_predict():

    sujet = get_ml_sujet_analyse()

    if (sujet == MLOPSCONST.ML_SUJET_ELEVE_RISQUE_ECHEC_4_PRIMAIRE
            or sujet == MLOPSCONST.ML_SUJET_ELEVE_RISQUE_ECHEC_6_PRIMAIRE):
        script_name = "predict_{}_{}.py".format(MLOPSCONST.ML_SUJET_ELEVE_RISQUE_ECHEC_PRIMAIRE, get_ml_methode_analyse())
    return script_name


def get_script_name_validate():
    script_name = "validate_data_{}_{}.py".format(get_ml_sujet_analyse(), get_ml_methode_analyse())
    return script_name


def get_script_name_split():
    sujet = get_ml_sujet_analyse()

    if (sujet == MLOPSCONST.ML_SUJET_ELEVE_RISQUE_ECHEC_4_PRIMAIRE
            or sujet == MLOPSCONST.ML_SUJET_ELEVE_RISQUE_ECHEC_6_PRIMAIRE):
        script_name = "split_data_{}_{}.py".format(MLOPSCONST.ML_SUJET_ELEVE_RISQUE_ECHEC_PRIMAIRE, get_ml_methode_analyse())
    return script_name


def get_script_name_remove_impute():
    sujet = get_ml_sujet_analyse()

    if (sujet == MLOPSCONST.ML_SUJET_ELEVE_RISQUE_ECHEC_4_PRIMAIRE
            or sujet == MLOPSCONST.ML_SUJET_ELEVE_RISQUE_ECHEC_6_PRIMAIRE):
        script_name = "remove_impute_data_{}_{}.py".format(MLOPSCONST.ML_SUJET_ELEVE_RISQUE_ECHEC_PRIMAIRE, get_ml_methode_analyse())
    return script_name


def check_envVars(envVars):
    for var in envVars:
        if var not in os.environ:
            print(f"### ERROR: Variable d'environmental '{var}' manquante !")
            exit(1)


def init_env_ere4p():
    env_path = Path('.') / 'config' / '.env_ere4p'

    load_dotenv(dotenv_path=env_path, override=False)

    check_envVars(['ML_SUJET', 'ML_METHODE', 'ML_TIMEOUT_PIPELINE', 'ML_TIMEOUT_SQL', 'ML_TIMEOUT_SQL_CONNEXION',
                   'ML_SQL_DATASTORE_NAME', 'ML_TIMEOUT_SLEEP_VALUE_CHECKER', 'ML_TIMEOUT_CLEAN_SUBMIT'])

    print("EDOTrace Variables environnement")
    print(os.environ['ML_SUJET'], os.environ['ML_METHODE'])


def init_env_ere6p():
    env_path = Path('.') / 'config' / '.env_ere6p'

    load_dotenv(dotenv_path=env_path, override=False)

    check_envVars(['ML_SUJET', 'ML_METHODE', 'ML_TIMEOUT_PIPELINE', 'ML_TIMEOUT_SQL', 'ML_TIMEOUT_SQL_CONNEXION',
                   'ML_SQL_DATASTORE_NAME', 'ML_TIMEOUT_SLEEP_VALUE_CHECKER', 'ML_TIMEOUT_CLEAN_SUBMIT'])

    print("EDOTrace Variables environnement")
    print(os.environ['ML_SUJET'], os.environ['ML_METHODE'])


def init_env(env: str):
    if (env == MLOPSCONST.ML_SUJET_ELEVE_RISQUE_ECHEC_4_PRIMAIRE):
        init_env_ere4p()
    elif (env == MLOPSCONST.ML_SUJET_ELEVE_RISQUE_ECHEC_6_PRIMAIRE):
        init_env_ere6p()
    else:
        raise Exception("EDOTrace Environnement non initialisé dans ml\\config")


def get_sql_connectstring(workspace: Workspace):

    connectstring = ""
    try:
        sql_datastore_exists, sql_datastore = get_sql_datastore(workspace)

        if sql_datastore_exists is False:
            raise UserErrorException("EDOTrace Le DataStore SQL doit être définit")

        sql_server_name = "{}{}".format(sql_datastore.server_name, '.database.windows.net')
        sql_database_name = sql_datastore.database_name
        sql_username = sql_datastore.username
        sql_userpassword = sql_datastore.password
        sql_driver = '{ODBC Driver 17 for SQL Server}'
        sql_connexion_timeout = "{}".format(get_ml_timeout_sql_connexion())

        connectstring = 'DRIVER=' + sql_driver + ';SERVER=' + sql_server_name + ';PORT=1433;DATABASE=' + \
            sql_database_name + ';UID=' + sql_username + ';PWD=' + sql_userpassword + ';TIMEOUT=' + sql_connexion_timeout

    except UserErrorException as error:
        print(error)

    return connectstring


def get_aml_compute_target(__file__):
    aml_compute_target_name = MLOPSCONST.AML_COMPUTE_TARGET_NAME

    workspace = get_workspace(__file__)

    try:
        aml_compute_target = AmlCompute(workspace=workspace, name=aml_compute_target_name)
    except ComputeTargetException:
        print("EDOTrace création d'une instance de calcul")

        provisioning_config = AmlCompute.provisioning_configuration(vm_size="STANDARD_D3_V2",
                                                                    min_nodes=0,
                                                                    max_nodes=4)

        aml_compute_target = ComputeTarget.create(workspace, aml_compute_target_name, provisioning_config)
        aml_compute_target.wait_for_completion(show_output=True, min_node_count=None, timeout_in_minutes=20)

    return aml_compute_target


def get_docker_base_image(workspace: Workspace):
    """
    manon

    # Retracer le nom du container registry et contruire le nom de l'image docker
    workspace.set_connection(
        name="privateAcr",
        category="ACR",
        target="<acr url>",
        authType="RegistryConnection",
        value={"ResourceId": "<user-assigned managed identity resource id>", "ClientId": "<user-assigned managed identity client ID>"})
    """
    acr_name = get_aml_acr_name(workspace)
    docker_base_image = "{}.{}/{}".format(acr_name, 'azurecr.io', MLOPSCONST.DOCKER_SPECIFICATIONS_IMAGE_BASE_FILE)
    return docker_base_image


def get_aml_environnement_nomad(workspace: Workspace, __file__) -> Environment:
    """ Python s'exécute dans des environnements virtuels (Conda, pip)
        Azure creates a Docker container and creates the environment.
    """

    conda_env_name = get_conda_env_name()

    conda_env_specification = get_conda_env_specification(__file__)

    env = Environment.from_conda_specification(name=conda_env_name, file_path=conda_env_specification)

    env.python.user_managed_dependencies = False

    env.docker.base_image = get_docker_base_image(workspace)
    env.docker.base_dockerfile = None
    env.version = MLOPSCONST.CONDA_ENV_VERSION

    # ----------------------------------------------------------
    # Initialisation des variables d'environnement.
    # Utilisé par les scripts pour déduire le sujet et la méthode
    # pour faire l'analyse prédictive.
    # --------------------------------------------
    # DEPRECATED DEPUI SDK 1.38 - maintenant on utilise run_config.environment_variables
    # env.environment_variables = get_conda_env_variables()

    # --------------------------------------------
    # Enregistrer officiellement notre environnement
    # --------------------------------------------
    env.register(workspace=workspace)

    env = Environment.get(workspace=workspace, name=conda_env_name)

    return env


def get_aml_environnement(workspace: Workspace, __file__) -> Environment:
    """ Python s'exécute dans des environnements virtuels (Conda, pip)
        Azure creates a Docker container and creates the environment.
    """

    conda_env_name = get_conda_env_name()

    conda_env_specification = get_conda_env_specification(__file__)

    env = Environment.from_conda_specification(name=conda_env_name, file_path=conda_env_specification)

    env.python.user_managed_dependencies = False

    # ---------------------------------------------------------------
    # Obtenir l'image de base enrichit par driver odbc pour MS SQL
    # ---------------------------------------------------------------
    docker_specification_fullpath = get_docker_specification_fullpath(__file__)

    with open(docker_specification_fullpath, "r") as f:
        dockerfile_contents_of_base_image = f.read()
        f.close()

    env.docker.base_image = None
    env.docker.base_dockerfile = dockerfile_contents_of_base_image
    env.version = MLOPSCONST.CONDA_ENV_VERSION

    # ----------------------------------------------------------
    # Initialisation des variables d'environnement.
    # Utilisé par les scripts pour déduire le sujet et la méthode
    # pour faire l'analyse prédictive.
    # --------------------------------------------
    # DEPRECATED DEPUI SDK 1.38 - maintenant on utilise run_config.environment_variables
    # env.environment_variables = get_conda_env_variables()

    # --------------------------------------------
    # Enregistrer officiellement notre environnement
    # --------------------------------------------
    env.register(workspace=workspace)

    env = Environment.get(workspace=workspace, name=conda_env_name)

    return env


# =============================
# Section Dataset name
# =============================


def get_datasetname_raw(sujet: str,
                        no_organisme: str,
                        prediction_pour_annee: int,
                        prediction_numero: str):
    """ Retourne le nom du train dataset format brute, tel que définit dans AML Workspace """
    datasetname = "{}_{}_{}_{}_{}".format(sujet, no_organisme, prediction_pour_annee, prediction_numero, "raw")
    return datasetname


def get_datasetname_train_raw(sujet: str,
                              no_organisme: str,
                              prediction_pour_annee: int,
                              prediction_numero: str):
    """ Retourne le nom du train dataset format brute, tel que définit dans AML Workspace """
    datasetname = "{}_{}_{}_{}_{}_{}".format(sujet, no_organisme, prediction_pour_annee, prediction_numero, "train", "raw")
    return datasetname


def get_datasetname_test_raw(sujet: str,
                             no_organisme: str,
                             prediction_pour_annee: int,
                             prediction_numero: str):
    """ Retourne le nom du test dataset format brute,  tel que définit dans
    AML Workspace.  Servira pour la prédiction-scoring. """
    datasetname = "{}_{}_{}_{}_{}_{}".format(sujet, no_organisme, prediction_pour_annee, prediction_numero, "test", "raw")
    return datasetname


def get_datasetname_subsettrain_raw(sujet: str,
                                    no_organisme: str,
                                    prediction_pour_annee: int,
                                    prediction_numero: str):
    """ Retourne le nom du train dataset contenant un échantillon réduit pour l'entrainement format brute. """
    datasetname = "{}_{}_{}_{}_{}_{}_{}".format(sujet, no_organisme, prediction_pour_annee, prediction_numero, "subset", "train", "raw")
    return datasetname


def get_datasetname_subsettest_raw(sujet: str,
                                   no_organisme: str,
                                   prediction_pour_annee: int,
                                   prediction_numero: str):
    """ Retourne le nom du test dataset contenant un échantillon réduit pour faire la prédiction format brute. """
    datasetname = "{}_{}_{}_{}_{}_{}_{}".format(sujet, no_organisme, prediction_pour_annee, prediction_numero, "subset", "test", "raw")
    return datasetname


def get_datasetname_train_prepared(sujet: str,
                                   no_organisme: str,
                                   prediction_pour_annee: int,
                                   prediction_numero: str):
    """ Retourne le nom du train dataset format préparé, tel que définit dans AML Workspace """
    datasetname = "{}_{}_{}_{}_{}_{}".format(sujet, no_organisme, prediction_pour_annee, prediction_numero, "train", "prepared")
    return datasetname


def get_datasetname_test_prepared(sujet: str,
                                  no_organisme: str,
                                  prediction_pour_annee: int,
                                  prediction_numero: str):
    """ Retourne le nom du test dataset format préparé,  tel que définit dans
    AML Workspace.  Servira pour la prédiction-scoring. """
    datasetname = "{}_{}_{}_{}_{}_{}".format(sujet, no_organisme, prediction_pour_annee, prediction_numero, "test", "prepared")
    return datasetname


def get_datasetname_subsettrain_prepared(sujet: str,
                                         no_organisme: str,
                                         prediction_pour_annee: int,
                                         prediction_numero: str):
    """ Retourne le nom du train dataset contenant un échantillon réduit pour l'entrainement format préparé. """
    datasetname = "{}_{}_{}_{}_{}_{}_{}".format(
        sujet, no_organisme, prediction_pour_annee, prediction_numero, "subset", "train", "prepared")
    return datasetname


def get_datasetname_subsettest_prepared(sujet: str,
                                        no_organisme: str,
                                        prediction_pour_annee: int,
                                        prediction_numero: str):
    """ Retourne le nom du test dataset contenant un échantillon réduit pour faire la prédiction format préparé. """
    datasetname = "{}_{}_{}_{}_{}_{}_{}".format(
        sujet, no_organisme, prediction_pour_annee, prediction_numero, "subset", "test", "prepared")
    return datasetname


def define_raw_dataset_input(workspace,
                             amlfolder,
                             sujet: str,
                             no_organisme: str,
                             prediction_pour_annee: int,
                             prediction_numero: str):
    """ Création du raw dataset dans l'espace AML """

    # -------------------------------
    # Construire le nom du dataset
    # -------------------------------
    dataset_file_name = "{}_{}_{}_{}_{}".format(sujet,
                                                no_organisme,
                                                prediction_pour_annee,
                                                prediction_numero,
                                                MLOPSCONST.RAWTYPE)
    # -------------------------------
    # Déterminer l'emplacement
    # -------------------------------
    datastore = workspace.get_default_datastore()

    datastore_paths = []

    datastore_path = (
        datastore,
        '{}{}.csv'.format(amlfolder, dataset_file_name)
    )

    datastore_paths.insert(0, datastore_path)

    dataset = Dataset.Tabular.from_delimited_files(path=datastore_paths,)

    # Puisque les dataset sont globaux, on doit préciser la nature de
    # l'information.  Contrairement au fichier référé qui lui est classé
    # dans des sous-répertoires.

    raw_datasetname = dataset_file_name

    # Register the defined dataset for later use
    dataset.register(
        workspace=workspace,
        name=raw_datasetname,
        description='',
        create_new_version=True,
    )


def get_root_folder_caller(__file__):
    """Permet de retracer le répertoire racine de l'appelant en assumant qu'on
       recule de 2 niveaux.
    """
    root_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..")
    return root_folder

# ----------------------------------
# Nouvel arrivage de données
# ----------------------------------


def get_folder_dataraw_incoming(__file__):
    root_folder = get_root_folder_caller(__file__)
    folder = os.path.join(root_folder, "data", "raw", "incoming")
    return folder


def get_amlfolder_dataraw_incoming():
    """ Retourne le chemin d'accès relatif dans l'expace AML.
    """
    amlfolder = "data/raw/incoming/"
    return amlfolder

# ----------------------------------
# Data en entrée à traiter
# ----------------------------------


def get_folder_dataraw_input(__file__):

    root_folder = get_root_folder_caller(__file__)
    folder = os.path.join(root_folder, "data", "raw", "input")
    return folder


def get_folder_dataraw_remove_impute(__file__):

    root_folder = get_root_folder_caller(__file__)
    folder = os.path.join(root_folder, "data", "raw", "remove_impute")
    return folder


def get_amlfolder_dataraw_input():
    """ Retourne le chemin d'accès relatif dans l'expace AML.
        Quand on regarde un dataset, on peut retracer le fichier csv associé.
        Ce fichier csv est dans le chemain d'accès relatif de l'espace AML.
        Exemple: data/raw/input/E R E 4 P_subset_test_CS1_E0.csv
        Ca retournera data/raw/input
    """
    amlfolder = "data/raw/input/"
    return amlfolder

# ----------------------------------
# Data prepared
# ----------------------------------


def get_folder_data_prep(__file__):
    root_folder = get_root_folder_caller(__file__)
    folder = os.path.join(root_folder, "data", "prep", "output")
    return folder


def get_folder_data_prep_train(__file__):
    root_folder = get_root_folder_caller(__file__)
    folder = os.path.join(root_folder, "data", "prep", "output", "train")
    return folder


def get_folder_data_prep_test(__file__):
    root_folder = get_root_folder_caller(__file__)
    folder = os.path.join(root_folder, "data", "prep", "output", "test")
    return folder


def get_folder_data_prep_subsettrain(__file__):
    root_folder = get_root_folder_caller(__file__)
    folder = os.path.join(root_folder, "data", "prep", "output", "subsettrain")
    return folder


def get_folder_data_prep_subsettest(__file__):
    root_folder = get_root_folder_caller(__file__)
    folder = os.path.join(root_folder, "data", "prep", "output", "subsettest")
    return folder


def get_folder_prediction_model(__file__):
    root_folder = get_root_folder_caller(__file__)
    folder = os.path.join(root_folder, "data", "prediction", "model")
    return folder


def get_folder_prediction_resultat(__file__):
    root_folder = get_root_folder_caller(__file__)
    folder = os.path.join(root_folder, "data", "prediction", "resultat")
    return folder


def get_folder_pipeline_output_images(__file__):
    root_folder = get_root_folder_caller(__file__)
    folder = os.path.join(root_folder, "data", "pipeline", "images")
    return folder


def get_folder_data_synthetic_output(__file__):
    root_folder = get_root_folder_caller(__file__)
    folder = os.path.join(root_folder, "data", "raw", "synthetic")

    return folder


def download_resultat_prediction_pour_imputation(workspace: Workspace,
                                                 __file__,
                                                 no_organisme: str,
                                                 annee_imputation: int,
                                                 prediction_numero: str,
                                                 experiment_name,
                                                 run_id: str):

    experiment = Experiment(workspace, experiment_name)
    pipeline_run = PipelineRun(experiment, run_id)

    step_run = pipeline_run.find_step_run(MLOPSCONST.PIPELINE_TP_STEP_PREDICT)[0]

    step_run_output = step_run.get_output("prediction_final")

    port_data_reference = step_run_output.get_port_data_reference()

    folder = get_folder_imputation_resultat(__file__)
    # -----------------------------------------------------------------
    # S'assurer de nettoyer le répertoire data\prediction\imputation
    # -----------------------------------------------------------------
    try:
        if os.path.exists(folder):
            shutil.rmtree(folder)

    except OSError as exception:
        print("Error: %s : %s" % (folder, exception.strerror))

    imputation_resultat_fullfilename = ''
    file_downloaded = port_data_reference.download(local_path=folder, overwrite=True)

    if file_downloaded:
        # ----------------------------------------------------------------------------------------------
        # Le fichier se retrouve dans un sous répertoire variable, le retracer en fonction
        # du run_id pour le step du pipeline.   Simplement retracer le fichier dans ce sous-répertoire.
        # ----------------------------------------------------------------------------------------------
        search_files = glob.glob(folder + "/**/azureml/**/*.csv", recursive=True)

        imputation_resultat_fullfilename = search_files[0]

        print("EDOTrace Fichier contenant le résultat de la prédiction 2020 est ici: ", imputation_resultat_fullfilename)
    else:
        print("EDOTrace Aucun fichier contenant le résultat de la prédiction 2020")

    return imputation_resultat_fullfilename


def download_pipeline_train_output(workspace: Workspace,
                                   __file__,
                                   no_organisme: str,
                                   annee_imputation: int,
                                   prediction_numero: str,
                                   experiment_name,
                                   run_id: str):

    experiment = Experiment(workspace, experiment_name)
    pipeline_run = PipelineRun(experiment, run_id)

    # --------------------------------------------------------
    # STEP 0 - Éliminer élèves doubleurs  / Imputation COVID
    # --------------------------------------------------------
    step_run = pipeline_run.find_step_run(MLOPSCONST.PIPELINE_TP_STEP_REMOVE_IMPUTE)[0]
    step_run_output = step_run.get_output("raw_remove_impute_data")
    port_data_reference = step_run_output.get_port_data_reference()
    folder = get_folder_dataraw_remove_impute(__file__)
    port_data_reference.download(local_path=folder, overwrite=True)

    # ------------------------------------------
    # Step 1 - Segregation (Split)
    # ------------------------------------------
    step_run = pipeline_run.find_step_run(MLOPSCONST.PIPELINE_TP_STEP_SEGREGATE)[0]

    step_run_output = step_run.get_output("raw_subset_train_data")
    port_data_reference = step_run_output.get_port_data_reference()
    folder = get_folder_data_raw_subsettrain(__file__)
    port_data_reference.download(local_path=folder, overwrite=True)

    step_run_output = step_run.get_output("raw_subset_test_data")
    port_data_reference = step_run_output.get_port_data_reference()
    folder = get_folder_data_raw_subsettest(__file__)
    port_data_reference.download(local_path=folder, overwrite=True)

    step_run_output = step_run.get_output("raw_train_data")
    port_data_reference = step_run_output.get_port_data_reference()
    folder = get_folder_data_raw_train(__file__)
    port_data_reference.download(local_path=folder, overwrite=True)

    step_run_output = step_run.get_output("raw_test_data")
    port_data_reference = step_run_output.get_port_data_reference()
    folder = get_folder_data_raw_test(__file__)
    port_data_reference.download(local_path=folder, overwrite=True)

    # ------------------------------------------
    # Step 2 - Preparation
    # ------------------------------------------
    step_run = pipeline_run.find_step_run(MLOPSCONST.PIPELINE_TP_STEP_PREPARE_DATA)[0]

    step_run_output = step_run.get_output("prep_train_data")
    port_data_reference = step_run_output.get_port_data_reference()
    folder = get_folder_data_prep_train(__file__)
    port_data_reference.download(local_path=folder, overwrite=True)

    step_run_output = step_run.get_output("prep_test_data")
    port_data_reference = step_run_output.get_port_data_reference()
    folder = get_folder_data_prep_test(__file__)
    port_data_reference.download(local_path=folder, overwrite=True)

    step_run_output = step_run.get_output("prep_subset_train_data")
    port_data_reference = step_run_output.get_port_data_reference()
    folder = get_folder_data_prep_subsettrain(__file__)
    port_data_reference.download(local_path=folder, overwrite=True)

    step_run_output = step_run.get_output("prep_subset_test_data")
    port_data_reference = step_run_output.get_port_data_reference()
    folder = get_folder_data_prep_subsettest(__file__)
    port_data_reference.download(local_path=folder, overwrite=True)

    # ------------------------------------------
    # Step 3 - Recherche Hyperparameter + Train
    # ------------------------------------------
    step_run = pipeline_run.find_step_run(MLOPSCONST.PIPELINE_TP_STEP_HYPER_TUNE_TRAIN)[0]

    step_run_output = step_run.get_output("prediction")
    port_data_reference = step_run_output.get_port_data_reference()
    folder = get_folder_prediction_resultat(__file__)
    port_data_reference.download(local_path=folder, overwrite=True)

    # ------------------------------------------
    # Step 4 - Prédiction
    # ------------------------------------------
    step_run = pipeline_run.find_step_run(MLOPSCONST.PIPELINE_TP_STEP_PREDICT)[0]

    step_run_output = step_run.get_output("prediction_final")
    port_data_reference = step_run_output.get_port_data_reference()
    folder = get_folder_prediction_resultat(__file__)
    port_data_reference.download(local_path=folder, overwrite=True)


def download_pipeline_output_images(workspace: Workspace,
                                    __file__,
                                    no_organisme: str,
                                    annee_imputation: int,
                                    prediction_numero: str,
                                    experiment_name,
                                    run_id: str):
    """
    Permet de télécharger le contenu de l'onglet Outputs+Logs du portail Azure Machine Learning.
    Du coup on va avoir les images localement.
    """

    experiment = Experiment(workspace, experiment_name)
    pipeline_run = PipelineRun(experiment, run_id)

    folder = get_folder_pipeline_output_images(__file__)

    folder_run_id = os.path.join(folder, run_id)

    pipeline_run.download_files(output_directory=folder_run_id)

    return folder_run_id


def download_pipeline_ingest_output(workspace: Workspace,
                                    __file__,
                                    no_organisme: str,
                                    annee_imputation: int,
                                    prediction_numero: str,
                                    experiment_name,
                                    run_id: str):

    experiment = Experiment(workspace, experiment_name)
    pipeline_run = PipelineRun(experiment, run_id)

    # --------------------------------------------------------
    # STEP 0 - Ingestion
    # --------------------------------------------------------
    step_run = pipeline_run.find_step_run(MLOPSCONST.PIPELINE_IN_STEP_INGEST)[0]
    step_run_output = step_run.get_output("NouvellesDonnees")
    port_data_reference = step_run_output.get_port_data_reference()
    folder = get_folder_dataraw_input(__file__)
    port_data_reference.download(local_path=folder, overwrite=True)

    # ----------------------------------------------------------
    # Reconstruire l'emplacement officiel du fichier téléchargé
    # ----------------------------------------------------------
    localfolder = "{}\\{}".format(folder, port_data_reference.path_on_datastore)

    print("localfolder", localfolder)

    return localfolder


def get_run_pipeline_train(workspace: Workspace,
                           __file__,
                           no_organisme: str,
                           annee_imputation: int,
                           prediction_numero: str,
                           experiment_name,
                           run_id: str):

    experiment = Experiment(workspace, experiment_name)
    pipeline_run = PipelineRun(experiment, run_id)
    return pipeline_run


def get_run_pipeline_ingest(workspace: Workspace,
                            __file__,
                            no_organisme: str,
                            annee_imputation: int,
                            prediction_numero: str,
                            experiment_name,
                            run_id: str):

    experiment = Experiment(workspace, experiment_name)
    pipeline_run = PipelineRun(experiment, run_id)
    return pipeline_run


def get_metrics_pipeline_train(workspace: Workspace,
                               __file__,
                               no_organisme: str,
                               annee_imputation: int,
                               prediction_numero: str,
                               experiment_name,
                               run_id: str):

    experiment = Experiment(workspace, experiment_name)
    pipeline_run = PipelineRun(experiment, run_id)

    # --------------------------------------------------------
    # Lecture des métriques du pipeline
    # --------------------------------------------------------
    metrics = pipeline_run.get_metrics(recursive=False)

    # ----------------------------------------
    # IMPORTANT Si recursive=False il faudra spécifier le run_id
    # ----------------------------------------
    # metrics = metrics[run_id]

    return metrics, pipeline_run


"""
Fabrication d'un dataframe de base pour lancer un pipeline
"""


def init_mlops_data(no_organisme: str, prediction_pour_annee: int, prediction_numero: str):
    mlops_data = [{'no_organisme': no_organisme,
                   'prediction_pour_annee': prediction_pour_annee,
                   'prediction_numero': prediction_numero,
                   'IN_nom_experience': '',
                   'IN_est_fait': '0',
                   'IN_code_sortie': '',
                   'IN_est_verifie': '0',
                   'IN_id_serie': '',
                   'IN_date_execution': '',
                   'TP_nom_experience': '',
                   'TP_est_fait': '0',
                   'TP_code_sortie': '',
                   'TP_est_verifie': '0',
                   'TP_id_serie': '',
                   'TP_date_execution': '',
                   'TP_est_verifie_par_assistant': '0',
                   }]

    mlops_dataframe = pd.DataFrame(mlops_data)

    return mlops_dataframe


"""
Pour COVID

Correspondra au fichier de prédiction à utiliser pour faire l'imputation.

Détails:
On invoque un pipeline pour faire la prédiction 2020 puis on télécharge le
résultat de la prédiction dans le dossier imputation.  Le fichier sera lu, pour faire l'imputation

"""


def get_folder_imputation_resultat(__file__):
    root_folder = get_root_folder_caller(__file__)
    folder = os.path.join(root_folder, "data", "prediction", "imputation")
    return folder


def create_folder(folder):
    """ Création du répertoire local dans lequel on génèrera des fichiers pour l'entrainement et la prédiction. """

    if not os.path.exists(folder):
        os.makedirs(folder)

    print("EDOTrace Création du répertoire local : " + folder)


def get_folder_data_raw(__file__):
    root_folder = get_root_folder_caller(__file__)
    folder = os.path.join(root_folder, "data", "raw", "output")
    return folder


def get_folder_data_raw_train(__file__):
    root_folder = get_root_folder_caller(__file__)
    folder = os.path.join(root_folder, "data", "raw", "output", "train")
    return folder


def get_folder_data_raw_test(__file__):
    root_folder = get_root_folder_caller(__file__)
    folder = os.path.join(root_folder, "data", "raw", "output", "test")
    return folder


def get_folder_data_raw_subsettrain(__file__):
    root_folder = get_root_folder_caller(__file__)
    folder = os.path.join(root_folder, "data", "raw", "output", "subsettrain")
    return folder


def get_folder_data_raw_subsettest(__file__):
    root_folder = get_root_folder_caller(__file__)
    folder = os.path.join(root_folder, "data", "raw", "output", "subsettest")
    return folder


# =============================
# Section Filename
# =============================


def get_filename_for_datatype(sujet, no_organisme, prediction_pour_annee, prediction_numero, datatype):
    """ Retourne le "NomFichier.csv" bien formé de toutes ses parties.
    Valeurs possibles pour datatype = train, test, subset_train, subset_test
    Exemple:  E R E 4 P_train_CS1_2019_E0.csv  Contient les données brutes d'entrainement 2012 à 2018
    et contient également les donnée brutes pour faire le scoring/prédiction de 2019 à l'étape 0
    """
    filename = "{}_{}_{}_{}_{}.csv".format(sujet, no_organisme, prediction_pour_annee, prediction_numero, datatype)

    return filename


def get_filename_for_labeldatatype(sujet, no_organisme, prediction_pour_annee, prediction_numero, datatype):
    """ Retourne le "NomFichier.csv" bien formé de toutes ses parties.
    Valeurs possibles pour datatype = train, test, subset_train, subset_test
    Exemple:  E R E 4 P_train_CS1_2019_E0_label.csv  Contient les données brutes d'entrainement 2012 à 2018
    et contient également les donnée brutes pour faire le scoring/prédiction de 2019 à l'étape 0
    """
    filename = "{}_{}_{}_{}_{}_{}.csv".format(sujet, no_organisme, prediction_pour_annee, prediction_numero, datatype, MLOPSCONST.LABEL)

    return filename


def get_filename_for_model(no_organisme, prediction_pour_annee, prediction_numero):

    sujet = get_ml_sujet_analyse()
    methode = get_ml_methode_analyse()

    filename = "{}_{}_{}_{}_{}_{}.sav".format(sujet, methode, no_organisme, prediction_pour_annee, prediction_numero, MLOPSCONST.ML_MODEL)
    return filename


def get_filename_for_standardisation(no_organisme, prediction_pour_annee, prediction_numero):
    """Nom de fichier pour la méthode de standardisation Robust Scaler"""
    sujet = get_ml_sujet_analyse()
    methode = get_ml_methode_analyse()

    filename = "{}_{}_{}_{}_{}_standardisation.pickle".format(sujet, methode, no_organisme, prediction_pour_annee, prediction_numero)
    return filename


def get_filename_for_standardisation_subset(no_organisme, prediction_pour_annee, prediction_numero):

    sujet = get_ml_sujet_analyse()
    methode = get_ml_methode_analyse()

    filename = "{}_{}_{}_{}_{}_standardisation_subset.pickle".format(sujet, methode, no_organisme, prediction_pour_annee, prediction_numero)
    return filename


def get_filename_for_best_classification_threshold(no_organisme, prediction_pour_annee, prediction_numero):

    sujet = get_ml_sujet_analyse()
    methode = get_ml_methode_analyse()

    filename = "{}_{}_{}_{}_{}_best_classification_threshold.pickle".format(sujet, methode, no_organisme, prediction_pour_annee, prediction_numero)
    return filename


def get_filename_for_model_subset(no_organisme, prediction_pour_annee, prediction_numero):

    sujet = get_ml_sujet_analyse()
    methode = get_ml_methode_analyse()

    filename = "{}_{}_{}_{}_{}_{}.sav".format(sujet, methode, no_organisme, prediction_pour_annee, prediction_numero, MLOPSCONST.ML_MODEL_SUBSET)
    return filename


def get_filename_for_prediction_resultat(no_organisme, prediction_pour_annee, prediction_numero):

    sujet = get_ml_sujet_analyse()
    methode = get_ml_methode_analyse()

    filename = "{}_{}_{}_{}_{}_{}.csv".format(sujet, methode, no_organisme, prediction_pour_annee, prediction_numero, MLOPSCONST.ML_PREDICTION)
    return filename


def get_filename_for_imputation_resultat(no_organisme, prediction_pour_annee, prediction_numero):

    sujet = get_ml_sujet_analyse()
    methode = get_ml_methode_analyse()

    filename = "{}_{}_{}_{}_{}_{}.csv".format(sujet, methode, no_organisme, prediction_pour_annee, prediction_numero, MLOPSCONST.ML_PREDICTION)
    return filename


def get_filename_for_prediction_resultat_subset(no_organisme, prediction_pour_annee, prediction_numero):

    sujet = get_ml_sujet_analyse()
    methode = get_ml_methode_analyse()

    filename = "{}_{}_{}_{}_{}_{}.csv".format(sujet, methode, no_organisme, prediction_pour_annee, prediction_numero, MLOPSCONST.ML_PREDICTION_SUBSET)
    return filename


def get_fullfilename_for_model(model_folder: str,
                               no_organisme: str,
                               prediction_pour_annee: int,
                               prediction_numero: str):
    filename = get_filename_for_model(no_organisme, prediction_pour_annee, prediction_numero)
    fullfilename = os.path.join(model_folder, filename)
    return fullfilename


def get_fullfilename_for_model_subset(model_folder: str,
                                      no_organisme: str,
                                      prediction_pour_annee: int,
                                      prediction_numero: str):
    filename = get_filename_for_model_subset(no_organisme, prediction_pour_annee, prediction_numero)
    fullfilename = os.path.join(model_folder, filename)
    return fullfilename


def get_fullfilename_for_standardisation(data_folder: str,
                                         no_organisme: str,
                                         prediction_pour_annee: int,
                                         prediction_numero: str):
    filename = get_filename_for_standardisation(no_organisme, prediction_pour_annee, prediction_numero)
    fullfilename = os.path.join(data_folder, filename)
    return fullfilename


def get_fullfilename_for_standardisation_subset(data_folder: str,
                                                no_organisme: str,
                                                prediction_pour_annee: int,
                                                prediction_numero: str):

    filename = get_filename_for_standardisation_subset(no_organisme, prediction_pour_annee, prediction_numero)
    fullfilename = os.path.join(data_folder, filename)
    return fullfilename


def get_fullfilename_for_best_classification_threshold(data_folder: str,
                                                       no_organisme: str,
                                                       prediction_pour_annee: int,
                                                       prediction_numero: str):
    filename = get_filename_for_best_classification_threshold(no_organisme, prediction_pour_annee, prediction_numero)
    fullfilename = os.path.join(data_folder, filename)
    return fullfilename


def get_fullfilename_for_prediction_resultat(resultat_folder: str,
                                             no_organisme: str,
                                             prediction_pour_annee: int,
                                             prediction_numero: str):
    filename = get_filename_for_prediction_resultat(no_organisme, prediction_pour_annee, prediction_numero)
    fullfilename = os.path.join(resultat_folder, filename)
    return fullfilename


def get_fullfilename_for_imputation_resultat(imputation_folder: str,
                                             no_organisme: str,
                                             prediction_pour_annee: int,
                                             prediction_numero: str):
    filename = get_filename_for_imputation_resultat(no_organisme, prediction_pour_annee, prediction_numero)
    fullfilename = os.path.join(imputation_folder, filename)
    return fullfilename


def get_fullfilename_for_prediction_resultat_subset(resultat_folder: str,
                                                    no_organisme: str,
                                                    prediction_pour_annee: int,
                                                    prediction_numero: str):
    filename = get_filename_for_prediction_resultat_subset(no_organisme, prediction_pour_annee, prediction_numero)
    fullfilename = os.path.join(resultat_folder, filename)
    return fullfilename


def get_fullfilename_dataraw_input(data_folder: str,
                                   sujet: str,
                                   no_organisme: str,
                                   prediction_pour_annee: int,
                                   prediction_numero: str):
    """ Retourne l'< emplacement > et < NomFichier.csv > bien formé de toutes ses parties.
        Ce fichier est considéré comme le résultat du pipeline d'ingestion.
        Exemple: data / raw / input / E R E 4 P_train_CS1_2019_E0.csv
    """
    filename = get_filename_for_datatype(sujet, no_organisme, prediction_pour_annee, prediction_numero, MLOPSCONST.RAWTYPE)

    fullfilename = os.path.join(data_folder, filename)

    return fullfilename


def get_fullfilename_dataraw_remove_impute(data_folder: str,
                                           sujet: str,
                                           no_organisme: str,
                                           prediction_pour_annee: int,
                                           prediction_numero: str):
    """ Retourne l'< emplacement > et < NomFichier.csv > bien formé de toutes ses parties.
        Ce fichier est considéré comme le résultat d'un nettoyage et d'une imputation dans la première étape
        du pipeline d'entraînement.
        Exemple: data / raw / remove_impute / E R E 4 P_CS1_2019_E0_raw_remove_impute.csv
    """
    filename = get_filename_for_datatype(sujet, no_organisme, prediction_pour_annee, prediction_numero, MLOPSCONST.RAWREMOVEIMPUTETYPE)

    fullfilename = os.path.join(data_folder, filename)

    return fullfilename


def get_fullfilename_dataraw_incoming(data_folder: str,
                                      sujet: str,
                                      no_organisme: str,
                                      prediction_pour_annee: int,
                                      prediction_numero: str):
    """ Retourne l'< emplacement > et < NomFichier.csv > bien formé de toutes ses parties.
        Ce fichier est considéré comme le résultat du pipeline d'ingestion.
        Exemple: data / raw / incoming / E R E 4 P_train_CS1_2019_E0.csv
    """
    filename = get_filename_for_datatype(sujet, no_organisme, prediction_pour_annee, prediction_numero, MLOPSCONST.RAWTYPE)

    fullfilename = os.path.join(data_folder, filename)

    return fullfilename


def get_fullfilename_dataraw_output(data_folder: str,
                                    sujet: str,
                                    datatype: str,
                                    no_organisme: str,
                                    prediction_pour_annee: int,
                                    prediction_numero: str):
    """ Retourne le nom complet du dossier et du fichier lorsque on génère un fichier
    d'entrainement et de prédiction en fonction d'un fichier d'entrée.
     Retourne l'< emplacement > et < NomFichier.csv > lorsque on génère un fichier
    d'entrainement et de prédiction en fonction d'un fichier d'entrée.
        Exemple: data / raw / output / E R E 4 P_train_CS1_2019_E0.csv
    """
    filename = get_filename_for_datatype(sujet, no_organisme, prediction_pour_annee, prediction_numero, datatype)

    fullfilename = os.path.join(data_folder, filename)

    return fullfilename


def get_fullfilename_data_prepared_output(data_folder,
                                          sujet: str,
                                          datatype: str,
                                          no_organisme: str,
                                          prediction_pour_annee: int,
                                          prediction_numero: str):
    """ Retourne le nom complet du dossier et du fichier transformé
    par la phase data_engineering.
    Exemple: data / prepared / output / E R E 4 P_train_CS1_2019_E0.csv
    """
    filename = get_filename_for_datatype(sujet, no_organisme, prediction_pour_annee, prediction_numero, datatype)

    filenamelabel = get_filename_for_labeldatatype(sujet, no_organisme, prediction_pour_annee, prediction_numero, datatype)

    fullfilename = os.path.join(data_folder, filename)
    fullfilenamelabel = os.path.join(data_folder, filenamelabel)

    return fullfilename, fullfilenamelabel

# ==================================
# Fonctions utilitaires pour l'entrainement et la prediction
# ==================================


def load_pickle(path):
    with open(path, 'rb') as handle:
        item = pickle.load(handle)
    return item


def save_pickle(fullfilename, item):
    pickle.dump(item, open(fullfilename, 'wb'))


def standardiser(train, test, features_ne_pas_standardiser=None, quantile_range=(0, 100)):
    """Standardiser les features non-binaires."""

    np.random.seed(13)

    robustScaler = RobustScaler(quantile_range=quantile_range)

    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    if features_ne_pas_standardiser is None:
        train = pd.DataFrame(robustScaler.fit_transform(train), columns=train.columns)
        test = pd.DataFrame(robustScaler.transform(test), columns=test.columns)
    else:
        trainStandardise = pd.DataFrame(robustScaler.fit_transform(train.drop(columns=features_ne_pas_standardiser)),
                                        columns=train.drop(columns=features_ne_pas_standardiser).columns)
        train = pd.concat([train[features_ne_pas_standardiser], trainStandardise], axis=1)

        testStandardise = pd.DataFrame(robustScaler.transform(test.drop(columns=features_ne_pas_standardiser)),
                                       columns=test.drop(columns=features_ne_pas_standardiser).columns)
        test = pd.concat([test[features_ne_pas_standardiser], testStandardise], axis=1)

    return train, test, robustScaler


def inverser_standardisation(dataframe, robustScaler, features_ne_pas_standardiser=None):
    """Inverser la standardisation des features non-binaires."""

    np.random.seed(13)

    if features_ne_pas_standardiser is None:
        dataframe = pd.DataFrame(robustScaler.inverse_transform(dataframe), columns=dataframe.columns)
    else:
        dataframeDestandardise = pd.DataFrame(robustScaler.inverse_transform(dataframe.drop(columns=features_ne_pas_standardiser)),
                                              columns=dataframe.drop(columns=features_ne_pas_standardiser).columns)

        dataframe = pd.concat([dataframe[features_ne_pas_standardiser], dataframeDestandardise], axis=1)

    return dataframe


def plot_autolabel_barh(main_plot, precision):
    """
    Attacher un label pour chaque rectangle.
    """

    for bar in main_plot.patches:
        width = bar.get_width()

        if width < 0:
            pos_x = width - 2.1
        else:
            pos_x = width + 0.2

        main_plot.text(
            x=pos_x,
            y=bar.get_y() + (bar.get_height() / 2.5),
            s="{:.{}f}".format(width, precision),
            color='dimgrey'
        )


def plot_autolabel_barv(main_plot, precision):
    """
    Attacher un label pour chaque rectangle.
    """

    for bar in main_plot.patches:
        height = bar.get_height()

        pos_y = height

        main_plot.text(
            x=bar.get_x() + (bar.get_width() / 2.5),
            y=pos_y,
            s="{:.{}f}".format(height, precision),
            color='dimgrey'
        )


def plot_performance(main_plot,
                     title,
                     run,
                     x_label=None,
                     y_label=None,
                     h_line=None,
                     h_line_label=None,
                     v_line=None,
                     v_line_label=None,
                     point=None,
                     point_label=None,
                     legend=False,
                     width=None,
                     height=None,
                     autolabel_barh=False,
                     autolabel_barv=False,
                     autolabel_precision=0,
                     show_title=True,
                     show_plot=False):
    """Afficher des graphiques en lien aux metriques de performance.

    Parametres
    ----------
        main_plot (plot): graphique principal servant de background
        title (string): titre du graphique
        run (azureml.core.run): classe utilitaire Azure ML indiquant si l'execution est locale ou dans le cloud
        x_label (string): nom de l'axe des x
        y_label (string): nom de l'axe des y
        h_line (float): emplacement d'une ligne horizontale superposee sur le graphique principal
        h_line_label (string): nom de la ligne horizontale
        v_line (float): emplacement d'une ligne verticale superposee sur le graphique principal
        v_line_label (string): nom de la ligne verticale
        point (float): emplacement d'un point superpose sur le graphique principal
        point_label (string): nom du point
        legend (bool): afficher la legende sur le graphique
        width (int): largeur du graphique
        height (int): hauteur du graphique
        show_plot (bool): plot le graphique directement
    """
    info = main_plot
    if h_line is not None:
        plt.plot(list(plt.xlim()), [h_line, h_line], linestyle='--', label=h_line_label)
    if v_line is not None:
        plt.plot([v_line, v_line], list(plt.ylim()), linestyle='--', label=v_line_label)
    if point is not None:
        plt.plot(point[0], point[1], 'ro', label=point_label)
    if legend:
        plt.legend()
    if x_label is not None:
        plt.xlabel(x_label)
    if y_label is not None:
        plt.ylabel(y_label)
    if width is not None or height is not None:
        plt.gcf().set_size_inches(width, height)
    if autolabel_barh is True:
        plot_autolabel_barh(main_plot, autolabel_precision)
    if autolabel_barv is True:
        plot_autolabel_barv(main_plot, autolabel_precision)
    if show_title:
        plt.title(title)
    plt.tight_layout()

    if show_plot:
        plt.show()

    if run:
        run.log_image(title, plot=plt.gcf())

    plt.close(plt.gcf())

    return info


def calculer_ratio_echec_reel_predit(nb_echec_reel, nb_echec_predit):
    if nb_echec_reel > 0:
        return nb_echec_predit * 100 / nb_echec_reel
    else:
        return 0


def identifier_zones_echec(opts,
                           EREP_colonne_zone_echec,
                           dataframe,
                           bins=[0, 59, 73, 100],
                           labels=["Zone 1\n0% à 59%", "Zone 2\n60% à 73%", "Zone 3\n74% à 100%"],
                           nom_colonne_echec='zone_echec',
                           preserver_seulement_colonnes_necessaires=False,
                           traitement_nulls=True):
    """Identifier a quelle zone d'echec appartient chaque eleve echoueur.

    On sépare les élèves en 3 catégories selon la note / RésultatEcriture en langue d'enseignement.
        Si on est à l'étape 0, on regarde le résultat écriture de 3e pour le classement
        Si on est à l'étape 1, on regarde le résultat écriture de 4e pour l'étape 1
        Si on est à l'étape 2, on regarde le résultat écriture de 4e pour l'étape 2

    On classe les élèves en 3 zones (3 bins)

        Un élève avec un résultatEcriture < 60% sera dans la catégorie Zone 1
        Un élève avec un résultatEcriture >= 60% et < 74% sera dans la catégorie Zone 2
        Un élève avec un résultatEcriture >= 74% et <= 100% sera dans la catégorie Zone 3
            """

    # --------------------------------------------------
    # Préparation données selon le moment de prédiction à faire
    # --------------------------------------------------
    moment_prediction = opts.prediction_numero
    col_res_selon_etape = EREP_colonne_zone_echec[moment_prediction]

    # Si la colonne RESULTAT_ECRITURE_ETAPE_1_4E_ANNEE ou RESULTAT_ECRITURE_ETAPE_2_4E_ANNEE a ete supprimee
    # durant la preparation des donnees, se replier sur les moments precedents
    if moment_prediction == '2' and col_res_selon_etape not in dataframe.columns:
        col_res_selon_etape = EREP_colonne_zone_echec['1']
        moment_prediction = '1'

    if moment_prediction == '1' and col_res_selon_etape not in dataframe.columns:
        col_res_selon_etape = EREP_colonne_zone_echec['0']

    # --------------------------------------------------
    # Repartition des zones d'echec
    # --------------------------------------------------
    dataframe[nom_colonne_echec] = pd.cut(
        dataframe[col_res_selon_etape],
        bins,
        labels=labels,
        right=True,
        include_lowest=True)

    # Remplacer les nulles par zone 1
    if traitement_nulls:
        dataframe.loc[dataframe[nom_colonne_echec].isnull(), nom_colonne_echec] = labels[0]

    # Distinguer les eleves en reussite
    dataframe[nom_colonne_echec].cat.add_categories(["Reussite"], inplace=True)
    dataframe.loc[dataframe['IndEchecEcritureExamen'] == 0, nom_colonne_echec] = "Reussite"

    # Preserver les colonnes strictement necessaires
    if preserver_seulement_colonnes_necessaires:
        dataframe = dataframe[[nom_colonne_echec, col_res_selon_etape, 'IndEchecEcritureExamen']]

    return dataframe


def performance_echoueurs_silencieux(opts, EREP_colonne_zone_echec, X_test, y_test, y_pred):
    """
    Graphique qu'on sort à l'étape de train #1 sur le jeux de validation lorsque on est dans le train script.
    Graphique qu'on sort à l'étape de prédiction si on est dans une année qu'on connait la réponse.

    Afin de voir qu'on trouve dans chaque zone le bon nombre d'élève
    Afin de voir les échoueurs silencieux
    On fait un graphique par zone pour montrer la répartition entre ce que la prédiction a donnée pour cet élève versus la réalité.
    On calcul le ratio Prédit / Réel * 100 par zone
    On calcul le ratio Prédit / Réel * 100 pour les 3 zones
    On calcul le ratio Prédiction globale
    """

    # --------------------------------------------------
    # Fabriquer les 3 zones selon les résultats.
    # --------------------------------------------------
    df = pd.concat([X_test, y_test], axis=1)
    df = identifier_zones_echec(opts, EREP_colonne_zone_echec, df, preserver_seulement_colonnes_necessaires=True, traitement_nulls=False)
    df = pd.concat([df, y_pred[0]], axis=1)
    df.columns = ['zone_echec', 'ResEcriture', 'EchecReel', 'EchecPredit']

    assert df.isnull().any().sum() == 0, "EDOTrace Situation imprévue détectée.  Null dans le dataset."

    # --------------------------------------------------
    # Conserver que les dossiers réellement en échec.
    # --------------------------------------------------
    df = df.loc[((df['EchecReel'] == 1))]

    # --------------------------------------------------
    # Calculer le ratio par zone
    # --------------------------------------------------
    df_sommaire = df.groupby('zone_echec')[['EchecReel', 'EchecPredit']].sum().reset_index()
    df_sommaire['Ratio'] = df_sommaire.apply(lambda row: calculer_ratio_echec_reel_predit(row['EchecReel'], row['EchecPredit']), axis=1)

    print("EDOTrace Tableau pour les échoueurs silencieux", df_sommaire)

    # --------------------------------------------------
    # Calculer le ratio global EchecReel vs EchecPredit peu importe la zone
    # --------------------------------------------------
    tot_reel = df_sommaire['EchecReel'].sum()
    tot_prediction = df_sommaire['EchecPredit'].sum()

    if tot_reel > 0:
        ratio_global = tot_prediction * 100 / tot_reel
    else:
        ratio_global = 0

    text_ratio_global = "Prédiction global\n{}%".format(format(ratio_global, '.1f'))

    # Dessiner le graphique
    ax = df_sommaire.plot(x="zone_echec", y=["EchecReel", "EchecPredit"], kind='bar')

    # -----------------------------------------------------------------
    # Nommer clairement les textes pour la légende et la positionner
    # ------------------------------------------------------------------
    ax.legend(['Réel', 'Prédit'], bbox_to_anchor=(1, 1), loc='upper left')

    # --------------------------------------------------------------------------------
    # Ajuster la hauteur du graphique selon la valeur max de y en ajoutant un 30%
    # --------------------------------------------------------------------------------
    ybottom, ytop = plt.ylim()
    plt.ylim(0, ytop * 1.3)

    # ---------------------------------------
    # Placer le texte en x horizontallement
    # ---------------------------------------
    plt.xticks(rotation=0)

    zone = 0

    for bar in ax.patches:
        ax.annotate(format(bar.get_height(), '.0f'),
                    (bar.get_x() + bar.get_width() / 2,
                        bar.get_height()), ha='center', va='center',
                    size=10, xytext=(0, 8),
                    textcoords='offset points')

        if zone < 3:

            text_groupe = "Prédiction Zone {}\n{}%".format(zone + 1, format(df_sommaire.iloc[zone].Ratio, '.1f'))

            # Extraire la limite en axe des y et ajuster la coordonnée pour voir l'annotation.
            ybottom, ytop = plt.ylim()
            coord_y = bar.get_height()

            ecart_vertical = 8

            if coord_y + ecart_vertical >= ytop:
                new_coord_y = coord_y - (ecart_vertical * 2)
            else:
                new_coord_y = coord_y + ecart_vertical

            ax.annotate(text_groupe,
                        (zone, new_coord_y),
                        ha='center',
                        va='bottom',
                        bbox=dict(boxstyle='round', facecolor='gray', alpha=0.1),
                        )

            zone += 1

    # --------------------------------------
    # Affichage du ratio global
    # --------------------------------------

    plt.text(1.27, 0.05,
             text_ratio_global,
             horizontalalignment='center',
             # verticalalignment='center',
             transform=ax.transAxes,
             bbox=dict(facecolor='gray', alpha=0.1),
             fontdict={'color': 'black', 'size': 12})

    return df_sommaire


def calculerMetriques(y_test, y_pred, print_results=True):
    """Calculer les metriques de performance."""
    confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
    assert len(confusion_matrix) == 2, "L'échantillon de test ne contient aucun échoueur."
    score1 = confusion_matrix[1, 0] / confusion_matrix.sum(axis=1)[1]
    score2 = confusion_matrix.sum(axis=0)[1] / confusion_matrix.sum()
    fpr = confusion_matrix[0, 1] / confusion_matrix.sum(axis=1)[0]
    score_pondere = ((score1 * 3) ** 2) + ((score2 * 1) ** 2)
    if print_results:
        print('Pourcentage des élèves réellement en échec non-capturés (score 1):', score1)
        print('Pourcentage des élèves prédits en échec (score 2):', score2)

    f1_weighted = metrics.f1_score(y_test, y_pred, average='weighted')
    mcc = metrics.matthews_corrcoef(y_test, y_pred)
    accuracy = metrics.accuracy_score(y_test, y_pred, normalize=True, sample_weight=None)
    balanced_accuracy = metrics.balanced_accuracy_score(y_test, y_pred)
    ratio_taux_pred_vs_reel = y_pred.mean() / y_test.mean()[0]

    return {'score1 (FNR)': score1,
            'score2': score2,
            'FPR': fpr,
            'score_pondere': score_pondere,
            'f1_weighted': f1_weighted,
            'MCC': mcc,
            'accuracy': accuracy,
            'balanced_accuracy': balanced_accuracy,
            'matrice de confusion': confusion_matrix,
            'ratio taux prediction vs reel': ratio_taux_pred_vs_reel}


def calculer_thresholds(modele, X_test, y_test):
    """Calculer les resultats de tests a divers thresholds."""
    test_scores = []
    preds_proba = modele.predict_proba(X_test)[:, 1]

    for threshold in np.arange(0, 1.01, 0.02):
        preds = (preds_proba > threshold) + 0
        scores = calculerMetriques(y_test, preds, print_results=False)
        scores['threshold'] = threshold
        test_scores.append(scores)
    test_scores = pd.DataFrame(test_scores)
    return test_scores


def plot_thresholds(test_scores, suptitle, best_threshold=None, num_ticks=10):
    """Afficher les resultats de test en fonction des thresholds."""
    num_threshold = len(test_scores)
    ticks_setting = (np.arange(num_ticks) / num_ticks) * num_threshold
    ticks_labels_empty = [None for i in np.arange(num_ticks)]
    ticks_labels = (np.arange(num_ticks) / num_ticks).round(3)

    fig, axs = plt.subplots(5, figsize=(7, 12))

    test_scores[['ratio taux prediction vs reel']].plot(ax=axs[0])
    axs[0].set_xticks(ticks_setting)
    axs[0].set_xticklabels(ticks_labels_empty)
    axs[0].fill_between((0, 50), 1, 1.5, facecolor='green', alpha=0.05)
    axs[0].plot((num_threshold / 2, num_threshold / 2), (0, 4), '--', linewidth=0.7, color='black')
    axs[0].set_ylim((1, 4))

    test_scores[['score1 (FNR)', 'score2', 'FPR']].plot(ax=axs[1])
    axs[1].set_xticks(ticks_setting)
    axs[1].set_xticklabels(ticks_labels_empty)
    axs[1].fill_between((0, 50), 0.05, 0.3, facecolor='green', alpha=0.05)
    axs[1].plot((num_threshold / 2, num_threshold / 2), (0, 1), '--', linewidth=0.7, color='black')
    axs[1].set_ylim((0.05, 0.5))

    test_scores[['balanced_accuracy']].plot(ax=axs[2])
    axs[2].set_xticks(ticks_setting)
    axs[2].set_xticklabels(ticks_labels_empty)
    axs[2].fill_between((0, 50), 0.7, test_scores['balanced_accuracy'].max() + 0.02, facecolor='green', alpha=0.05)
    axs[2].plot((num_threshold / 2, num_threshold / 2), (0, 1), '--', linewidth=0.7, color='black')
    axs[2].set_ylim((test_scores['balanced_accuracy'].min(), test_scores['balanced_accuracy'].max() + 0.02))

    test_scores[['f1_weighted']].plot(ax=axs[3])
    axs[3].set_xticks(ticks_setting)
    axs[3].set_xticklabels(ticks_labels_empty)
    axs[3].plot((num_threshold / 2, num_threshold / 2), (0, 1), '--', linewidth=0.7, color='black')
    axs[3].set_ylim((test_scores['f1_weighted'].min(), test_scores['f1_weighted'].max() + 0.02))

    test_scores[['MCC']].plot(ax=axs[4])
    axs[4].set_xticks(ticks_setting)
    axs[4].set_xticklabels(ticks_labels)
    axs[4].fill_between((0, 50), 0.3, test_scores['MCC'].max() + 0.02, facecolor='green', alpha=0.05)
    axs[4].plot((num_threshold / 2, num_threshold / 2), (0, 1), '--', linewidth=0.7, color='black')
    axs[4].set_ylim((0, test_scores['MCC'].max() + 0.02))

    if best_threshold is not None:
        location = num_threshold * best_threshold
        axs[0].plot((location, location), (0, 4), '--', linewidth=0.7, color='red')
        axs[1].plot((location, location), (0, 1), '--', linewidth=0.7, color='red')
        axs[2].plot((location, location), (0, 1), '--', linewidth=0.7, color='red')
        axs[3].plot((location, location), (0, 1), '--', linewidth=0.7, color='red')
        axs[4].plot((location, location), (0, 1), '--', linewidth=0.7, color='red')

    plt.xlabel('Threshold')
    fig.suptitle(suptitle)

# ==================================
# Utilitaire
# ==================================


def root_run(self: Run) -> Run:
    run = Run.get_context()

    if run._run_id.startswith("OfflineRun"):
        return None

    if self.parent is None:
        return self
    return root_run(self.parent)


def get_workspace(__file__, _file_name=None):
    """Retourne une référence vers le workspace azure machine learning en lien avec le
       fichier de configuration.  Le fichier se trouve à la racine du projet dans
       le dossier .azureml
    """
    run = Run.get_context()

    if run._run_id.startswith("OfflineRun"):
        run_offline_ref = True
    else:
        run_offline_ref = False

    try:
        if run_offline_ref:
            root_folder = get_root_folder_caller(__file__)
            # --------------------------------------------------------------------
            # Attention : Si ça plante pour une raison obscur sur votre poste...
            # Une des raison serait un changement de mot de passe.  Dans ce cas
            # Refaire une authentification avec : az login
            # En prod ce n'est pas un problème car on utilise un Service Principal.
            # --------------------------------------------------------------------
            workspace = Workspace.from_config(path=root_folder, auth=AzureCliAuthentication(), _file_name=_file_name)
        else:
            workspace = run.experiment.workspace

    except UserErrorException:
        workspace = run.experiment.workspace

    return workspace


def get_sql_datastore(workspace: Workspace):

    sql_datastore = None

    try:
        sql_datastore = Datastore.get(workspace, get_ml_sql_datastore_name())
        datastore_sql_exists = True

    except HttpOperationError:
        datastore_sql_exists = False

    except UserErrorException:
        datastore_sql_exists = False

    return datastore_sql_exists, sql_datastore


def get_vpd_prediction_pour_annee():
    now = datetime.datetime.now()
    if (now.month >= 7 and now.month <= 12):
        vpd_annee = now.year
    else:
        vpd_annee = now.year - 1

    return vpd_annee


def get_all_config_client(workspace: Workspace,
                          sujet: str):
    ''' Obtention de la configuration client pour le machine learning
    '''
    configClient = None

    # ---------------------------------------------------------
    # Vérifier si on a un DataStoreSQL et extraire les données.
    # ---------------------------------------------------------
    sql_datastore_exists, sql_datastore = get_sql_datastore(workspace)

    if (sql_datastore_exists):
        # ---------------------------------------------------------
        # Obtenir la configuration pour ce client
        # ---------------------------------------------------------
        print("EDOTrace requête SQL obtention configuration des clients en cours")

        if sujet == MLOPSCONST.ML_SUJET_ELEVE_RISQUE_ECHEC_4_PRIMAIRE:

            list_col_str = ','.join(map(str, MLOPSCONST.ERE4P_config_client_columns))

            the_query = "select {} From ConfigClient.EleveARisque".format(
                list_col_str)

            query = DataPath(sql_datastore, the_query)

            tabular = Dataset.Tabular.from_sql_query(
                query=query,
                validate=False,
                query_timeout=get_ml_timeout_sql(),
                set_column_types=MLOPSCONST.ERE4P_config_client_DataTypes)

            configClient = tabular.to_pandas_dataframe()
    else:
        raise Exception("EDOTrace Le datastore SQL est inexistant au niveau du workspace Azure ML")

        """
        vpd_prediction_pour_annee = get_vpd_prediction_pour_annee()

        # vpd_DatePrevu1erePrediction = datetime.date(vpd_prediction_pour_annee, 9, 1).strftime(MLOPSCONST.DATE_FORMAT_STR)
        # vpd_DatePrevu2ePrediction = datetime.date(vpd_prediction_pour_annee, 12, 15).strftime(MLOPSCONST.DATE_FORMAT_STR)
        # vpd_DatePrevu3ePrediction = datetime.date(vpd_prediction_pour_annee + 1, 4, 15).strftime(MLOPSCONST.DATE_FORMAT_STR)

        vpd_DatePrevu1erePrediction = datetime.date(vpd_prediction_pour_annee, 9, 1)
        vpd_DatePrevu2ePrediction = datetime.date(vpd_prediction_pour_annee, 12, 15)
        vpd_DatePrevu3ePrediction = datetime.date(vpd_prediction_pour_annee + 1, 4, 15)

        vpd_config_client = {'IdClient': '######',
                             'Annee': '####',
                             'CutOffPrediction': 50,
                             'ScoreFNR': 25,
                             'ScoreTPE': 35,
                             'DatePrevu1erePrediction': vpd_DatePrevu1erePrediction,
                             'DatePrevu2ePrediction': vpd_DatePrevu2ePrediction,
                             'DatePrevu3ePrediction': vpd_DatePrevu3ePrediction
                             }

        # creating a Dataframe object
        configClient = pd.DataFrame(vpd_config_client, index=[0])
        """

    print("EDOTrace configClient=", configClient)

    return configClient


def get_config_client(workspace: Workspace,
                      sujet: str,
                      no_organisme: str,
                      prediction_pour_annee: int,):
    ''' Obtention de la configuration client pour le machine learning
    '''
    configClient = None

    # ---------------------------------------------------------
    # Vérifier si on a un DataStoreSQL et extraire les données.
    # ---------------------------------------------------------
    sql_datastore_exists, sql_datastore = get_sql_datastore(workspace)

    if (sql_datastore_exists):
        # ---------------------------------------------------------
        # Obtenir la configuration pour ce client
        # ---------------------------------------------------------
        print("EDOTrace requête SQL obtention configuration client en cours")

        if sujet == MLOPSCONST.ML_SUJET_ELEVE_RISQUE_ECHEC_4_PRIMAIRE:

            list_col_str = ','.join(map(str, MLOPSCONST.ERE4P_config_client_columns))

            the_query = "select {} From ConfigClient.EleveARisque where IdClient={} and Annee={}".format(list_col_str, no_organisme, prediction_pour_annee)

            query = DataPath(sql_datastore, the_query)

            tabular = Dataset.Tabular.from_sql_query(
                query=query,
                validate=False,
                query_timeout=get_ml_timeout_sql(),
                set_column_types=MLOPSCONST.ERE4P_config_client_DataTypes)

            configClient = tabular.to_pandas_dataframe()
    else:
        vpd_DatePrevu1erePrediction = datetime.date(prediction_pour_annee, 9, 1).strftime(MLOPSCONST.DATE_FORMAT_STR)
        vpd_DatePrevu2ePrediction = datetime.date(prediction_pour_annee, 12, 15).strftime(MLOPSCONST.DATE_FORMAT_STR)
        vpd_DatePrevu3ePrediction = datetime.date(prediction_pour_annee + 1, 4, 15).strftime(MLOPSCONST.DATE_FORMAT_STR)

        vpd_config_client = {'CutOffPrediction': 50,
                             'ScoreFNR': 25,
                             'ScoreTPE': 35,
                             'DatePrevu1erePrediction': vpd_DatePrevu1erePrediction,
                             'DatePrevu2ePrediction': vpd_DatePrevu2ePrediction,
                             'DatePrevu3ePrediction': vpd_DatePrevu3ePrediction
                             }

        # creating a Dataframe object
        configClient = pd.DataFrame(vpd_config_client, index=[0])

    print("EDOTrace configClient=", configClient)

    return configClient


def delete_dataraw_input_from_local(__file__):
    '''
    Cette fonction va supprimer les fichiers csv locaux .
    '''

    localfolder = get_folder_dataraw_input(__file__)

    dir_path = localfolder

    print("---------------------------------------")
    print("supprimer fichiers locaux : " + dir_path)
    print("---------------------------------------")

    try:
        shutil.rmtree(dir_path)
    except OSError as exception:
        print("Error: %s : %s" % (dir_path, exception.strerror))


def autodeploy_remove(__file__):

    root_folder = get_root_folder_caller(__file__)

    dst_ingest_fullpath = os.path.join(root_folder, "code", "ingest", "ml")
    dst_remove_impute_fullpath = os.path.join(root_folder, "code", "remove_impute", "ml")
    dst_split_fullpath = os.path.join(root_folder, "code", "split", "ml")
    dst_prepare_fullpath = os.path.join(root_folder, "code", "prepare", "ml")
    dst_train_fullpath = os.path.join(root_folder, "code", "train", "ml")
    dst_predict_fullpath = os.path.join(root_folder, "code", "predict", "ml")
    dst_validate_fullpath = os.path.join(root_folder, "code", "validate", "ml")

    dst_ingest = "ml/code/ingest/ml/code/helper/script_helper.py"
    dst_remove_impute = "ml/code/remove_impute/ml/code/helper/script_helper.py"
    dst_split = "ml/code/split/ml/code/helper/script_helper.py"
    dst_prepare = "ml/code/prepare/ml/code/helper/script_helper.py"
    dst_train = "ml/code/train/ml/code/helper/script_helper.py"
    dst_predict = "ml/code/predict/ml/code/helper/script_helper.py"
    dst_validate = "ml/code/validate/ml/code/helper/script_helper.py"

    if os.path.exists(dst_ingest):
        os.chmod(dst_ingest, S_IWRITE)
        shutil.rmtree(dst_ingest_fullpath)

    if os.path.exists(dst_remove_impute_fullpath):
        os.chmod(dst_remove_impute, S_IWRITE)
        shutil.rmtree(dst_remove_impute_fullpath)

    if os.path.exists(dst_split):
        os.chmod(dst_split, S_IWRITE)
        shutil.rmtree(dst_split_fullpath)

    if os.path.exists(dst_prepare):
        os.chmod(dst_prepare, S_IWRITE)
        shutil.rmtree(dst_prepare_fullpath)

    if os.path.exists(dst_train):
        os.chmod(dst_train, S_IWRITE)
        shutil.rmtree(dst_train_fullpath)

    if os.path.exists(dst_predict):
        os.chmod(dst_predict, S_IWRITE)
        shutil.rmtree(dst_predict_fullpath)

    if os.path.exists(dst_validate):
        os.chmod(dst_validate, S_IWRITE)
        shutil.rmtree(dst_validate_fullpath)


def invoke_pipeline_ingest(
        workspace: Workspace,
        invoke_method,
        mlops_data,
        fullfilename_data_mlops_plan,
        max_experiments: int,
        monitor_pipeline_ingest: str):

    # -------------------------------------------------
    # Obtenir le PipelineEndPoint officiel d'ingestion
    # -------------------------------------------------
    pipeline_endpoint_name = get_pipeline_endpoint_name_ingest()
    pipelineEndpoint = PipelineEndpoint.get(workspace=workspace, name=pipeline_endpoint_name)

    # ---------------------------------
    # Début du traitement
    # ---------------------------------
    nb_experiments_inprogress = 0

    for index, row in mlops_data.iterrows():

        # ------------------------------
        # Est-ce déjà traité?
        # ------------------------------
        if row.IN_est_fait == '1':
            continue

        nb_experiments_inprogress = nb_experiments_inprogress + 1

        # ----------------------------------------------------------
        # Construire le nom de l'expérience pour donner du contexte
        # ----------------------------------------------------------
        experiment_name = get_experiment_name_ingestion(
            get_ml_sujet_analyse(),
            row.no_organisme,
            row.prediction_pour_annee,
            row.prediction_numero)

        # ------------------------------------------------------------------
        # Invocation du pipeline à travers le PipelineEndPoint Rest API
        # ------------------------------------------------------------------
        if invoke_method == MLOPSCONST.INVOKE_METHOD_REST:
            auth = InteractiveLoginAuthentication()
            aad_token = auth.get_authentication_header()
            rest_endpoint1 = pipelineEndpoint._endpoint

            response = requests.post(rest_endpoint1,
                                     headers=aad_token,
                                     json={"ExperimentName": experiment_name,
                                           "RunSource": "SDK",
                                           "ParameterAssignments": {"no_organisme": row.no_organisme,
                                                                    "prediction_pour_annee": row.prediction_pour_annee,
                                                                    "prediction_numero": row.prediction_numero,
                                                                    "pipeline_arg_incoming_folder": "data/raw/incoming",
                                                                    "pipeline_arg_input_folder": "data/raw/input/"}})
            try:
                response.raise_for_status()
            except Exception:
                raise Exception("EDOTrace Réception d'une mauvaise réponse du endpoint: {}\n"
                                "Response Code: {}\n"
                                "Headers: {}\n"
                                "Content: {}".format(rest_endpoint1, response.status_code, response.headers, response.content))
        else:
            pipelineRun = pipelineEndpoint.submit(experiment_name=experiment_name,
                                                  pipeline_parameters={"no_organisme": row.no_organisme,
                                                                       "prediction_pour_annee": row.prediction_pour_annee,
                                                                       "prediction_numero": row.prediction_numero,
                                                                       "pipeline_arg_incoming_folder": "data/raw/incoming",
                                                                       "pipeline_arg_input_folder": "data/raw/input/"}
                                                  )

            if monitor_pipeline_ingest != "1":
                pipelineRun.wait_for_completion(show_output=False, raise_on_error=True)

        mlops_data.at[index, 'IN_nom_experience'] = experiment_name
        mlops_data.at[index, 'IN_est_fait'] = '0'
        mlops_data.at[index, 'IN_code_sortie'] = '0'
        mlops_data.at[index, 'IN_est_verifie'] = '0'

        if invoke_method == MLOPSCONST.INVOKE_METHOD_REST:
            mlops_data.at[index, 'IN_id_serie'] = response.json().get('Id')
        else:
            mlops_data.at[index, 'IN_id_serie'] = pipelineRun.id

        # -----------------------------------------------------------------------------------------------------
        # On limite le nombre d'expériences concurentes afin de ne pas accaparer toutes les ressources Azure.
        # -----------------------------------------------------------------------------------------------------
        if nb_experiments_inprogress == max_experiments:

            if monitor_pipeline_ingest == "1":
                monitor_completion_ingest(workspace, mlops_data, fullfilename_data_mlops_plan)
                nb_experiments_inprogress = 0
            else:
                print("Nombre maximum d'experiences en concurrence atteint:{}", max_experiments)
                break

    # ----------------------------------------------------
    # Monitorer les dernières expériences restantes ...
    # ----------------------------------------------------
    if nb_experiments_inprogress > 0:

        if monitor_pipeline_ingest == "1":
            monitor_completion_ingest(workspace, mlops_data, fullfilename_data_mlops_plan)
            nb_experiments_inprogress = 0

    return mlops_data


def monitor_completion_ingest(workspace, data_to_monitor, fullfilename_data_mlops_plan):

    for index, row in data_to_monitor.iterrows():
        # ------------------------------
        # Est-ce déjà traité?
        # ------------------------------
        if row.IN_est_fait == '1':
            continue

        if pd.isnull(row.IN_nom_experience):
            continue

        print("EDOTrace Supervision de l'exécution du pipeline\n"
              "Expérience = {}\n"
              "Run_id = {}".format(row.IN_nom_experience, row.IN_id_serie))

        experiment = Experiment(workspace, row.IN_nom_experience)

        run = Run(experiment, row.IN_id_serie)

        # ------------------------------------------------
        # Combien de secondes avant de vérifier le statut
        # ------------------------------------------------
        timeout_sleep_value_checker = get_ml_timeout_sleep_value_checker()

        # -----------------------------------------------------------------
        # -----------------------------------------------------------------
        sleep_value_max_iteration = get_ml_sleep_value_max_iteration()

        while True:

            sleep(timeout_sleep_value_checker - time() % timeout_sleep_value_checker)

            details = run.get_detailed_status()
            status = details.get('status')

            if status == 'Completed':
                date_execution_str = datetime.datetime.now().strftime(MLOPSCONST.DATETIME_SHORT_FORMAT_STR)
                print("EDOTrace Le run_id={} est terminé à {}".format(row.IN_id_serie, date_execution_str))

                data_to_monitor.loc[data_to_monitor.IN_id_serie == row.IN_id_serie, 'IN_est_fait'] = '1'
                data_to_monitor.loc[data_to_monitor.IN_id_serie == row.IN_id_serie, 'IN_code_sortie'] = MLOPSCONST.PIPELINE_STATUS_COMPLETED
                data_to_monitor.loc[data_to_monitor.IN_id_serie == row.IN_id_serie, 'IN_date_execution'] = date_execution_str

                if fullfilename_data_mlops_plan != "":
                    data_to_monitor.to_csv(fullfilename_data_mlops_plan, index=False)
                break

            # -------------------------------------------------------
            # S'assurer qu'on ne roule pas jusqu'à la fin des temps
            # -------------------------------------------------------
            sleep_value_max_iteration = sleep_value_max_iteration - 1

            if sleep_value_max_iteration <= 0:
                raise Exception("EDOTrace Timeout.  Augmenter sleep_value_checker_timeout ou investiguer pourquoi via AML Workspace")

            if status == 'Running':
                continue
            elif status == 'Failed':
                date_execution_str = datetime.datetime.now().strftime(MLOPSCONST.DATETIME_SHORT_FORMAT_STR)
                print("EDOTrace Le run_id={} est en erreur à {} Veuiller investiguer dans le portail AML".format(row.IN_id_serie, date_execution_str))

                data_to_monitor.loc[data_to_monitor.IN_id_serie == row.IN_id_serie, 'IN_est_fait'] = '1'
                data_to_monitor.loc[data_to_monitor.IN_id_serie == row.IN_id_serie, 'IN_code_sortie'] = MLOPSCONST.PIPELINE_STATUS_ERROR
                data_to_monitor.loc[data_to_monitor.IN_id_serie == row.IN_id_serie, 'IN_date_execution'] = date_execution_str

                # ------------------------------------------
                # Sauvegarder les résultats sur l'avancement de l'ingestion
                # ------------------------------------------
                if fullfilename_data_mlops_plan != "":
                    data_to_monitor.to_csv(fullfilename_data_mlops_plan, index=False)
                break
            elif status == 'Canceled':
                date_execution_str = datetime.datetime.now().strftime(MLOPSCONST.DATETIME_SHORT_FORMAT_STR)
                print("EDOTrace Le run_id={} est en erreur à {} Veuiller investiguer dans le portail AML".format(row.IN_id_serie, date_execution_str))

                data_to_monitor.loc[data_to_monitor.IN_id_serie == row.IN_id_serie, 'IN_est_fait'] = '0'
                data_to_monitor.loc[data_to_monitor.IN_id_serie == row.IN_id_serie, 'IN_code_sortie'] = MLOPSCONST.PIPELINE_STATUS_ANNULE
                data_to_monitor.loc[data_to_monitor.IN_id_serie == row.IN_id_serie, 'IN_date_execution'] = date_execution_str

                if fullfilename_data_mlops_plan != "":
                    data_to_monitor.to_csv(fullfilename_data_mlops_plan, index=False)
                break

            print("EDOTrace en cours\n", details)

    if fullfilename_data_mlops_plan != "":
        data_to_monitor.to_csv(fullfilename_data_mlops_plan, index=False)

    return data_to_monitor


def invoke_pipeline_train(workspace: Workspace,
                          invoke_method,
                          mlops_data,
                          fullfilename_data_mlops_plan,
                          max_experiments: int,
                          monitor_pipeline_train: str,
                          prediction_pour_imputation_covid='0',
                          dataset_name_overwrite=''):

    # ----------------------------------------------------
    # Retracer le pipeline d'entrainement
    # ----------------------------------------------------
    pipeline_endpoint_name = get_pipeline_endpoint_name_train()
    pipelineEndpoint = PipelineEndpoint.get(workspace, name=pipeline_endpoint_name)

    # ---------------------------------
    # Début du traitement
    # ---------------------------------
    nb_experiments_inprogress = 0

    for index, row in mlops_data.iterrows():

        # ------------------------------
        # Est-ce déjà traité?
        # ------------------------------
        if row.TP_est_fait == '1':
            continue

        # ------------------------------
        # Sommes-nous dans un pipeline qui lance un pipeline (Cas COVID) ou circonstance normal ?
        #
        # -----------------------------------------------------------------------------------------
        if prediction_pour_imputation_covid == '0':
            if row.IN_est_fait == '0' or (row.IN_est_fait == '1' and row.IN_code_sortie != MLOPSCONST.PIPELINE_STATUS_COMPLETED):
                continue

        nb_experiments_inprogress = nb_experiments_inprogress + 1

        # ----------------------------------------------------------
        # Construire le nom de l'expérience pour donner du contexte
        # ----------------------------------------------------------
        experiment_name = get_experiment_name_train_pred(
            get_ml_sujet_analyse(),
            row.no_organisme,
            row.prediction_pour_annee,
            row.prediction_numero)

        if dataset_name_overwrite:
            dataset_name = dataset_name_overwrite
        else:
            dataset_name = get_datasetname_raw(
                get_ml_sujet_analyse(),
                row.no_organisme,
                row.prediction_pour_annee,
                row.prediction_numero)

        dataset = Dataset.get_by_name(workspace, dataset_name)
        print("dataset.id=", dataset.id)

        if invoke_method == MLOPSCONST.INVOKE_METHOD_REST:
            # ------------------------------------------------------------------
            # Invocation du pipeline à travers le PipelineEndPoint Rest API
            # ------------------------------------------------------------------
            auth = InteractiveLoginAuthentication()

            aad_token = auth.get_authentication_header()

            rest_endpoint1 = pipelineEndpoint._endpoint

            response = requests.post(rest_endpoint1,
                                     headers=aad_token,
                                     json={"ExperimentName": experiment_name,
                                           "RunSource": "SDK",
                                           "DataSetDefinitionValueAssignments": {"jeux_donnees": {"SavedDataSetReference": {"Id": dataset.id}}},
                                           "ParameterAssignments": {"no_organisme": row.no_organisme,
                                                                    "prediction_pour_annee": row.prediction_pour_annee,
                                                                    "prediction_numero": row.prediction_numero,
                                                                    "prediction_pour_imputation_covid": prediction_pour_imputation_covid,
                                                                    "pipeline_arg_incoming_folder": "data/raw/incoming",
                                                                    "pipeline_arg_input_folder": "data/raw/input/"}})

            try:
                response.raise_for_status()
            except Exception:
                raise Exception("EDOTrace Réception d'une mauvaise réponse du endpoint: {}\n"
                                "Response Code: {}\n"
                                "Headers: {}\n"
                                "Content: {}".format(rest_endpoint1, response.status_code, response.headers, response.content))
        else:
            pipelineRun = pipelineEndpoint.submit(experiment_name=experiment_name,
                                                  pipeline_parameters={"jeux_donnees": dataset,
                                                                       "no_organisme": row.no_organisme,
                                                                       "prediction_pour_annee": row.prediction_pour_annee,
                                                                       "prediction_numero": row.prediction_numero,
                                                                       "prediction_pour_imputation_covid": prediction_pour_imputation_covid,
                                                                       "pipeline_arg_incoming_folder": "data/raw/incoming",
                                                                       "pipeline_arg_input_folder": "data/raw/input/"}
                                                  )
            if monitor_pipeline_train != "1":
                pipelineRun.wait_for_completion(show_output=False, raise_on_error=True)

        mlops_data.at[index, 'TP_nom_experience'] = experiment_name
        mlops_data.at[index, 'TP_est_fait'] = '0'
        mlops_data.at[index, 'TP_code_sortie'] = '0'
        mlops_data.at[index, 'TP_est_verifie'] = '0'
        mlops_data.at[index, 'TP_est_verifie_par_assistant'] = ''  # Mis a '' au lieu de '0', car champ vide par default

        if invoke_method == MLOPSCONST.INVOKE_METHOD_REST:
            mlops_data.at[index, 'TP_id_serie'] = response.json().get('Id')
        else:
            mlops_data.at[index, 'TP_id_serie'] = pipelineRun.id

        # -----------------------------------------------------------------------------------------------------
        # On limite le nombre d'expériences concurentes afin de ne pas accaparer toutes les ressources Azure.
        # -----------------------------------------------------------------------------------------------------
        if nb_experiments_inprogress == max_experiments:
            if monitor_pipeline_train == "1":
                monitor_completion_train(workspace, mlops_data, fullfilename_data_mlops_plan)
                nb_experiments_inprogress = 0
            else:
                print("Nombre maximum d'experiences en concurrence atteint:{}", max_experiments)
                break

    # ----------------------------------------------------
    # Monitorer les dernières expériences restantes ...
    # ----------------------------------------------------
    if nb_experiments_inprogress > 0:

        if monitor_pipeline_train == "1":
            monitor_completion_train(workspace, mlops_data, fullfilename_data_mlops_plan)
            nb_experiments_inprogress = 0

    return mlops_data


def monitor_completion_train(workspace, data_to_monitor, fullfilename_data_mlops_plan):

    for index, row in data_to_monitor.iterrows():
        # ------------------------------
        # Est-ce déjà traité?
        # ------------------------------
        if row.TP_est_fait == '1':
            continue

        if pd.isnull(row.TP_nom_experience):
            continue

        print("EDOTrace Supervision de l'exécution du pipeline\n"
              "Expérience = {}\n"
              "Run_id = {}".format(row.TP_nom_experience, row.TP_id_serie))

        experiment = Experiment(workspace, row.TP_nom_experience)

        run = Run(experiment, row.TP_id_serie)

        # ------------------------------------------------
        # Combien de secondes avant de vérifier le statut
        # ------------------------------------------------
        timeout_sleep_value_checker = get_ml_timeout_sleep_value_checker()

        # -----------------------------------------------------------------
        # -----------------------------------------------------------------
        sleep_value_max_iteration = get_ml_sleep_value_max_iteration()

        while True:

            sleep(timeout_sleep_value_checker - time() % timeout_sleep_value_checker)

            details = run.get_detailed_status()
            status = details.get('status')

            if status == 'Completed':
                date_execution_str = datetime.datetime.now().strftime(MLOPSCONST.DATETIME_SHORT_FORMAT_STR)
                print("EDOTrace Le run_id={} est terminé à {}".format(row.TP_id_serie, date_execution_str))

                data_to_monitor.loc[data_to_monitor.TP_id_serie == row.TP_id_serie, 'TP_est_fait'] = '1'
                data_to_monitor.loc[data_to_monitor.TP_id_serie == row.TP_id_serie, 'TP_code_sortie'] = MLOPSCONST.PIPELINE_STATUS_COMPLETED
                data_to_monitor.loc[data_to_monitor.TP_id_serie == row.TP_id_serie, 'TP_date_execution'] = date_execution_str

                # ------------------------------------------------------------------------------
                # Sauvegarder les résultats sur l'avancement
                # Dans le cas de l'invocation COVID pour imputation nous n'avons pas de fichier
                # ------------------------------------------------------------------------------
                if fullfilename_data_mlops_plan != "":
                    data_to_monitor.to_csv(fullfilename_data_mlops_plan, index=False)
                break

            # -------------------------------------------------------
            # S'assurer qu'on ne roule pas jusqu'à la fin des temps
            # -------------------------------------------------------
            sleep_value_max_iteration = sleep_value_max_iteration - 1

            if sleep_value_max_iteration <= 0:
                raise Exception("EDOTrace Timeout.  Augmenter sleep_value_checker_timeout ou investiguer pourquoi via AML Workspace")

            if status == 'Running':
                continue
            elif status == 'Failed':
                date_execution_str = datetime.datetime.now().strftime(MLOPSCONST.DATETIME_SHORT_FORMAT_STR)
                print("EDOTrace Le run_id={} est en erreur à {} Veuiller investiguer dans le portail AML".format(row.TP_id_serie, date_execution_str))

                data_to_monitor.loc[data_to_monitor.TP_id_serie == row.TP_id_serie, 'TP_est_fait'] = '1'
                data_to_monitor.loc[data_to_monitor.TP_id_serie == row.TP_id_serie, 'TP_code_sortie'] = MLOPSCONST.PIPELINE_STATUS_ERROR
                data_to_monitor.loc[data_to_monitor.TP_id_serie == row.TP_id_serie, 'TP_date_execution'] = date_execution_str

                # ------------------------------------------
                # Sauvegarder les résultats sur l'avancement de l'ingestion
                # ------------------------------------------
                if fullfilename_data_mlops_plan != "":
                    data_to_monitor.to_csv(fullfilename_data_mlops_plan, index=False)
                break

            elif status == 'Canceled':
                date_execution_str = datetime.datetime.now().strftime(MLOPSCONST.DATETIME_SHORT_FORMAT_STR)
                print("EDOTrace Le run_id={} a été annulé à {} Veuiller investiguer dans le portail AML".format(row.TP_id_serie, date_execution_str))

                data_to_monitor.loc[data_to_monitor.TP_id_serie == row.TP_id_serie, 'TP_est_fait'] = '1'
                data_to_monitor.loc[data_to_monitor.TP_id_serie == row.TP_id_serie, 'TP_code_sortie'] = MLOPSCONST.PIPELINE_STATUS_ANNULE
                data_to_monitor.loc[data_to_monitor.TP_id_serie == row.TP_id_serie, 'TP_date_execution'] = date_execution_str

                # ------------------------------------------
                # Sauvegarder les résultats sur l'avancement de l'ingestion
                # ------------------------------------------
                if fullfilename_data_mlops_plan != "":
                    data_to_monitor.to_csv(fullfilename_data_mlops_plan, index=False)
                break

            print("EDOTrace en cours\n", details)

    if fullfilename_data_mlops_plan != "":
        data_to_monitor.to_csv(fullfilename_data_mlops_plan, index=False)


def save_prediction_to_sql_ERE6P(cursor, df_resultat):

    # ---------------------------------------------------------------------
    # Conversion des NaN en objet pour que SQL traite bien les valeurs null
    # ---------------------------------------------------------------------
    df_resultat_nullable = df_resultat.astype(object).where(pd.notnull(df_resultat), None)

    list_col_str = ','.join(map(str, MLOPSCONST.ERE6P_columns))

    the_query = "INSERT INTO AI.PredictionEREP ({}) ".format(list_col_str)

    for index, row in df_resultat_nullable.iterrows():
        cursor.execute(the_query + " values(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,\
            ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,\
            ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                       row.IdClient,
                       row.AnneePrediction,
                       row.Precision,
                       row.Organisme,
                       row.Fiche,
                       row.Annee,
                       row.Sexe,
                       row.GroupeRepere,
                       row.Age30Septembre,
                       row.IndFrancophone,
                       row.IndDoubleurAnneePrecedente,
                       row.IndDoubleurAnneeCourante,
                       row.NbAnneesRedoublees,
                       row.IndRedoublement,
                       row.RepondantPere,
                       row.RepondantMere,
                       row.RepondantTuteur,
                       row.InterdictionPere,
                       row.InterdictionMere,
                       row.InterdictionTuteur,
                       row.DecesPere,
                       row.DecesMere,
                       row.ScolaritePere,
                       row.ScolariteMere,
                       row.LieuNaissance,
                       row.CategorieLinguistique,
                       row.StatutGeneration,
                       row.Ecole,
                       row.ResEtape1Ecrire,
                       row.ResEtape2Ecrire,
                       row.ResEtape1Lire,
                       row.ResEtape2Lire,
                       row.ResEtape1Communiquer,
                       row.ResEtape2Communiquer,
                       row.ResEtape1Raisonner,
                       row.ResEtape2Raisonner,
                       row.ResEtape1Resoudre,
                       row.ResEtape2Resoudre,
                       row.ResEtape1CommuniquerLangueSeconde,
                       row.ResEtape2CommuniquerLangueSeconde,
                       row.ResEtape1LireLangueSeconde,
                       row.ResEtape2LireLangueSeconde,
                       row.ResEtape1EcrireLangueSeconde,
                       row.ResEtape2EcrireLangueSeconde,
                       row.NbAbsencesMotiveesEtape1,
                       row.NbAbsencesNonMotiveesEtape1,
                       row.NbRetardsEtape1,
                       row.NbAbsencesMotiveesEtape2,
                       row.NbAbsencesNonMotiveesEtape2,
                       row.NbRetardsEtape2,
                       row.IndPlanIntervention,
                       row.IndEHDAA,
                       row.IndMesureAide,
                       row.IMSERangDecile,
                       row.NbEcolesFrequentees,
                       row.ResFinal5eCommuniquer,
                       row.ResFinal5eLire,
                       row.ResFinal5eEcrire,
                       row.ResFinal5eResoudre,
                       row.ResFinal5eRaisonner,
                       row.ResFinal5eCommuniquerLangueSeconde,
                       row.ResFinal5eLireLangueSeconde,
                       row.ResFinal5eEcrireLangueSeconde,
                       row.Res5eEtape1Ecrire,
                       row.Res5eEtape2Ecrire,
                       row.Res5eEtape3Ecrire,
                       row.Res5eEtape1Lire,
                       row.Res5eEtape2Lire,
                       row.Res5eEtape3Lire,
                       row.Res5eEtape1Communiquer,
                       row.Res5eEtape2Communiquer,
                       row.Res5eEtape3Communiquer,
                       row.Res5eEtape1Raisonner,
                       row.Res5eEtape2Raisonner,
                       row.Res5eEtape3Raisonner,
                       row.Res5eEtape1Resoudre,
                       row.Res5eEtape2Resoudre,
                       row.Res5eEtape3Resoudre,
                       row.Res5eEtape1CommuniquerLangueSeconde,
                       row.Res5eEtape2CommuniquerLangueSeconde,
                       row.Res5eEtape3CommuniquerLangueSeconde,
                       row.Res5eEtape1LireLangueSeconde,
                       row.Res5eEtape2LireLangueSeconde,
                       row.Res5eEtape3LireLangueSeconde,
                       row.Res5eEtape1EcrireLangueSeconde,
                       row.Res5eEtape2EcrireLangueSeconde,
                       row.Res5eEtape3EcrireLangueSeconde,
                       row.NbAbsencesMotivees5eEtape1,
                       row.NbAbsencesNonMotivees5eEtape1,
                       row.NbRetards5eEtape1,
                       row.NbAbsencesMotivees5eEtape2,
                       row.NbAbsencesNonMotivees5eEtape2,
                       row.NbRetards5eEtape2,
                       row.NbAbsencesMotivees5eEtape3,
                       row.NbAbsencesNonMotivees5eEtape3,
                       row.NbRetards5eEtape3,
                       row.IndPlanIntervention5e,
                       row.IndEHDAA5e,
                       row.IndMesureAide5e,
                       row.IMSERangDecile5e,
                       row.NbEcolesFrequentees5e,
                       row.ResFinal4eCommuniquer,
                       row.ResFinal4eLire,
                       row.ResFinal4eEcrire,
                       row.ResFinal4eResoudre,
                       row.ResFinal4eRaisonner,
                       row.ResFinal4eCommuniquerLangueSeconde,
                       row.ResFinal4eLireLangueSeconde,
                       row.ResFinal4eEcrireLangueSeconde,
                       row.Res4eEtape1Ecrire,
                       row.Res4eEtape2Ecrire,
                       row.Res4eEtape3Ecrire,
                       row.Res4eEtape1Lire,
                       row.Res4eEtape2Lire,
                       row.Res4eEtape3Lire,
                       row.Res4eEtape1Communiquer,
                       row.Res4eEtape2Communiquer,
                       row.Res4eEtape3Communiquer,
                       row.Res4eEtape1Raisonner,
                       row.Res4eEtape2Raisonner,
                       row.Res4eEtape3Raisonner,
                       row.Res4eEtape1Resoudre,
                       row.Res4eEtape2Resoudre,
                       row.Res4eEtape3Resoudre,
                       row.Res4eEtape1CommuniquerLangueSeconde,
                       row.Res4eEtape2CommuniquerLangueSeconde,
                       row.Res4eEtape3CommuniquerLangueSeconde,
                       row.Res4eEtape1LireLangueSeconde,
                       row.Res4eEtape2LireLangueSeconde,
                       row.Res4eEtape3LireLangueSeconde,
                       row.Res4eEtape1EcrireLangueSeconde,
                       row.Res4eEtape2EcrireLangueSeconde,
                       row.Res4eEtape3EcrireLangueSeconde,
                       row.NbAbsencesMotivees4eEtape1,
                       row.NbAbsencesNonMotivees4eEtape1,
                       row.NbRetards4eEtape1,
                       row.NbAbsencesMotivees4eEtape2,
                       row.NbAbsencesNonMotivees4eEtape2,
                       row.NbRetards4eEtape2,
                       row.NbAbsencesMotivees4eEtape3,
                       row.NbAbsencesNonMotivees4eEtape3,
                       row.NbRetards4eEtape3,
                       row.IndPlanIntervention4e,
                       row.IndEHDAA4e,
                       row.IndMesureAide4e,
                       row.ResExamen4e,
                       row.ResFinal3eCommuniquer,
                       row.ResFinal3eLire,
                       row.ResFinal3eEcrire,
                       row.ResFinal3eResoudre,
                       row.ResFinal3eRaisonner,
                       row.ResFinal3eCommuniquerLangueSeconde,
                       row.ResFinal3eLireLangueSeconde,
                       row.ResFinal3eEcrireLangueSeconde,
                       row.NbAbsencesMotivees3eEtape1,
                       row.NbAbsencesNonMotivees3eEtape1,
                       row.NbRetards3eEtape1,
                       row.NbAbsencesMotivees3eEtape2,
                       row.NbAbsencesNonMotivees3eEtape2,
                       row.NbRetards3eEtape2,
                       row.NbAbsencesMotivees3eEtape3,
                       row.NbAbsencesNonMotivees3eEtape3,
                       row.NbRetards3eEtape3,
                       row.IndPlanIntervention3e,
                       row.IndEHDAA3e,
                       row.IndMesureAide3e,
                       row.NotePassageExamenEcriture,
                       row.ResExamen,
                       row.IndEchecEcritureExamen,
                       row.ModelePredictif,
                       row.TypeDataset,
                       row.Prediction,
                       row.PredictionProbabilite
                       )


def autodeploy():

    src = "ml/code/helper/script_helper.py"

    dst_ingest = "ml/code/ingest/ml/code/helper/script_helper.py"
    dst_remove_impute = "ml/code/remove_impute/ml/code/helper/script_helper.py"
    dst_split = "ml/code/split/ml/code/helper/script_helper.py"
    dst_prepare = "ml/code/prepare/ml/code/helper/script_helper.py"
    dst_train = "ml/code/train/ml/code/helper/script_helper.py"
    dst_predict = "ml/code/predict/ml/code/helper/script_helper.py"
    dst_validate = "ml/code/validate/ml/code/helper/script_helper.py"

    if os.path.exists(dst_ingest):
        os.chmod(dst_ingest, S_IWRITE)
    else:
        os.makedirs(os.path.dirname(dst_ingest), exist_ok=True)

    if os.path.exists(dst_remove_impute):
        os.chmod(dst_remove_impute, S_IWRITE)
    else:
        os.makedirs(os.path.dirname(dst_remove_impute), exist_ok=True)

    if os.path.exists(dst_split):
        os.chmod(dst_split, S_IWRITE)
    else:
        os.makedirs(os.path.dirname(dst_split), exist_ok=True)

    if os.path.exists(dst_prepare):
        os.chmod(dst_prepare, S_IWRITE)
    else:
        os.makedirs(os.path.dirname(dst_prepare), exist_ok=True)

    if os.path.exists(dst_train):
        os.chmod(dst_train, S_IWRITE)
    else:
        os.makedirs(os.path.dirname(dst_train), exist_ok=True)

    if os.path.exists(dst_predict):
        os.chmod(dst_predict, S_IWRITE)
    else:
        os.makedirs(os.path.dirname(dst_predict), exist_ok=True)

    if os.path.exists(dst_validate):
        os.chmod(dst_validate, S_IWRITE)
    else:
        os.makedirs(os.path.dirname(dst_validate), exist_ok=True)

    shutil.copy(src, dst_ingest)
    shutil.copy(src, dst_remove_impute)
    shutil.copy(src, dst_split)
    shutil.copy(src, dst_prepare)
    shutil.copy(src, dst_train)
    shutil.copy(src, dst_predict)
    shutil.copy(src, dst_validate)

    os.chmod(dst_ingest, S_IREAD)
    os.chmod(dst_remove_impute, S_IREAD)
    os.chmod(dst_split, S_IREAD)
    os.chmod(dst_prepare, S_IREAD)
    os.chmod(dst_train, S_IREAD)
    os.chmod(dst_predict, S_IREAD)
    os.chmod(dst_validate, S_IREAD)

    print("Déploiement script_helper terminé")


if __name__ == "__main__":
    # -----------------------------------------------
    # Lecture des paramètres pour l'exécution
    # ------------------------------------------------
    opts = parse_myargs()

#    if opts.autodeploy == '1':
    autodeploy()
    # autodeploy_remove(__file__)
