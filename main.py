import os
import telebot
import numpy as np
import pandas as pd
import random
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
from scipy import stats
from telebot import types
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import io

import heapq

import csv

from lifelines.utils import concordance_index
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index as cindex

#Аргоноплазмова коагуляція (Класична)  (APC - 30 ВТ. 0.8 л/хв) - Argon_plasma_coagulation_(Classic) - 1
#Аргоноплазмова коагуляція гібридна ( Hibrid APC) ( 30 Вт, 0.8л/хв - NaCl 0.9%  + Індігокарміну 0.2 %) - 2
#Аргоноплазмова коагуляція гібридна ( Hibrid APC) ( 30 Вт, 0.8л/хв - Гелоспан 4%  + Індігокарміну 0.2 %) - 3
#ESD + Аргоноплазмова коагуляція гібридна ( Hibrid APC) ( 30 Вт, 0.8л/хв - Гелоспан 4%  + Індігокарміну 0.2 %) - 4
#EMR + ESD Аргоноплазмова коагуляція гібридна ( Hibrid APC) ( 30 Вт, 0.8л/хв - NaCl 0.9%  + Індігокарміну 0.2 %) - 5

Gender = {'Жінка': 0, 'Чоловік': 1}
g_0 = str(list(Gender.keys())[0])
g_1 = str(list(Gender.keys())[1])

Obesity_dic = {'Ні': 0, 'Так': 1}
o_0 = str(list(Obesity_dic.keys())[0])
o_1 = str(list(Obesity_dic.keys())[1])

Type2Diabetes_dic = {'Ні': 0, 'Так': 1 }
d_0 = str(list(Type2Diabetes_dic.keys())[0])
d_1 = str(list(Type2Diabetes_dic.keys())[1])

Treatment_type_of_endoscopic_intervention = {'Аргоноплазмова коагуляція_Класична_APC - 30 ВТ. 0.8 л_хв': 1,
                            'Аргоноплазмова коагуляція гібридна_Hibrid APC_30 Вт 0.8л_хв_NaCl 0.9_Індігокарміну 0.2': 2,
                            'Аргоноплазмова коагуляція гібридна_Hibrid APC_30 Вт 0.8л_хв_Гелоспан 4_Індігокарміну 0.2': 3,
                            'ESD + Аргоноплазмова коагуляція гібридна ( Hibrid APC) ( 30 Вт, 0.8л/хв - Гелоспан 4%  + Індігокарміну 0.2 %)': 4,
                            'EMR + ESD Аргоноплазмова коагуляція гібридна ( Hibrid APC) ( 30 Вт, 0.8л/хв - NaCl 0.9%  + Індігокарміну 0.2 %)': 5
                            }
EI_0 = str(list(Treatment_type_of_endoscopic_intervention.keys())[0])
EI_1 = str(list(Treatment_type_of_endoscopic_intervention.keys())[1])
EI_2 = str(list(Treatment_type_of_endoscopic_intervention.keys())[2])
EI_3 = str(list(Treatment_type_of_endoscopic_intervention.keys())[3])
EI_4 = str(list(Treatment_type_of_endoscopic_intervention.keys())[4])


#Задня крурорафія + Фундоплікація за Ніссеном -1

Treatment_type_of_surgical_intervention = {'': 0,
                            'Задня крурорафія + Фундоплікація за Ніссеном': 1,
                            'Задня крурорафія + Фундоплікація за Тупе': 2
                            }

Type_of_Histological_conclusion = {'Без метаплазії та дисплазії': 0,
                                'З кишковою метаплазією без дисплазії': 1,
                                'З кишковою метаплазією з легкою дисплазією_LGD': 2,
                                'З кишковою метаплазією з важкою дисплазією_НGD': 3
                                }
HC_0 = str(list(Type_of_Histological_conclusion.keys())[0])
HC_1 = str(list(Type_of_Histological_conclusion.keys())[1])
HC_2 = str(list(Type_of_Histological_conclusion.keys())[2])
HC_3 = str(list(Type_of_Histological_conclusion.keys())[3])

RefluxEsophagitisTypes = {'Немає': 0, 'LA-A': 1, 'LA-B': 2, 'LA-C': 3, 'LA-D': 4}
RF_0 = str(list(RefluxEsophagitisTypes.keys())[0])
RF_1 = str(list(RefluxEsophagitisTypes.keys())[1])
RF_2 = str(list(RefluxEsophagitisTypes.keys())[2])
RF_3 = str(list(RefluxEsophagitisTypes.keys())[3])
RF_4 = str(list(RefluxEsophagitisTypes.keys())[4])

#global RecomendedTreatmentType_dic
RecomendedTreatmentType_dic = {}
RecomendedTreatmentType_coeffs_dic = {}

global age
age = None
global TypeOfHernia
TypeOfHernia = None
global Prague_Сlassification_M
Prague_Сlassification_M = None
global Level_GSOD_M
Level_GSOD_M = None
global RefluxEsophagitis
RefluxEsophagitis = None


np.random.seed(18)
random.seed(1)

patient_dict = {}
Barrettpatient_dict = {}

def RandomForestComplicationsProbabilityFunc(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10):
  # Завантаження даних у Pandas DataFrame
  #df = pd.read_csv(io.StringIO(csv_data))

  df = pd.read_csv('BarrettTreatmentComplication.csv')

  # Визначення ознак (X) та цільової змінної (y)
  # Ми прогнозуємо стовпець 'Ускладнення'
  df['RefluxEsophagitis'] = df['RefluxEsophagitis'].map(RefluxEsophagitisTypes)
  X = df.drop(['Level_GSOD_C', 'Prague_Сlassification_С','Surgical_inter', 'Complication'], axis=1)
  y = df['Complication']

  # Розділення даних на тренувальний та тестовий набори
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

  # Ініціалізація та навчання моделі (використовуємо Random Forest для класифікації)
  model = RandomForestClassifier(random_state=42)
  model.fit(X_train, y_train)

  # Прогнозування на тестовому наборі
  y_pred = model.predict(X_test)

  # Тепер ви можете використовувати навчену модель для прогнозування ускладнень для нових пацієнтів
  # Наприклад, для нового пацієнта з такими характеристиками:
  NewPatient = pd.DataFrame({
   'Age': [x1],
   'Sex': [x2],
   'RefluxEsophagitis': [x3],
   'TypeofHernia': [x4],
   'Type2Diabetes': [x5],
   'Obesity': [x6],
   'Prague_Сlassification_M': [x7],
   'Level_GSOD_M': [x8],
   'TypeofHistological_conclusion': [x9],
   'Endoscopic_inter': [x10]
  })

  #Resume_predicted_proba_lr_Cognitive_Disorders = Resume_predicted_proba_lr_Cognitive_Disorders[-1][1]

  ComplicationsProbability = model.predict(NewPatient)
  if ComplicationsProbability < 1:
    ComplicationsProbabilityAnswer = 'не очікується'
  else:
    ComplicationsProbabilityAnswer = 'очікується'
  ComplicationsProbabilityPercent = model.predict_proba(NewPatient)
  ComplicationsProbabilityPercent = ComplicationsProbabilityPercent[-1][1]
  ComplicationsProbabilityPercent = ComplicationsProbabilityPercent*100

  features = ['Age','Sex','RefluxEsophagitis','TypeofHernia','Type2Diabetes','Obesity','Prague_Сlassification_M','Level_GSOD_M','TypeofHistological_conclusion','Endoscopic_inter']
  importances = model.feature_importances_
  feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
  feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
  feature_importance_dict = feature_importance_df.set_index('Feature')['Importance'].to_dict()

  return feature_importance_dict, ComplicationsProbabilityAnswer, ComplicationsProbabilityPercent

# де 0 означає відсутність ускладнень, а 1 - наявність

def LogisticRegressionComplicationsProbabilityFunc(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10):
#LogisticRegression для прогнозування Ускладнення
  # Завантаження даних у Pandas DataFrame
  #df = pd.read_csv(io.StringIO(csv_data))

  df = pd.read_csv('BarrettTreatmentComplication.csv')

  # Визначення ознак (X) та цільової змінної (y)
  # Ми прогнозуємо стовпець 'Ускладнення'
  df['RefluxEsophagitis'] = df['RefluxEsophagitis'].map(RefluxEsophagitisTypes)
  X = df.drop(['Level_GSOD_C', 'Prague_Сlassification_С','Surgical_inter', 'Complication'], axis=1)
  y = df['Complication']

  # Розділення даних на тренувальний та тестовий набори
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


  # Ініціалізація та навчання моделі (використовуємо Random Forest для класифікації)
  model = LogisticRegression(random_state=42)
  model.fit(X_train, y_train)

  # Прогнозування на тестовому наборі
  y_pred = model.predict(X_test)

  # Тепер ви можете використовувати навчену модель для прогнозування ускладнень для нових пацієнтів
  # Наприклад, для нового пацієнта з такими характеристиками:
  NewPatient = pd.DataFrame({
   'Age': [x1],
   'Sex': [x2],
   'RefluxEsophagitis': [x3],
   'TypeofHernia': [x4],
   'Type2Diabetes': [x5],
   'Obesity': [x6],
   'Prague_Сlassification_M': [x7],
   'Level_GSOD_M': [x8],
   'TypeofHistological_conclusion': [x9],
   'Endoscopic_inter': [x10]
  })

  #Resume_predicted_proba_lr_Cognitive_Disorders = Resume_predicted_proba_lr_Cognitive_Disorders[-1][1]

  LogComplicationsProbability = model.predict(NewPatient)
  if LogComplicationsProbability < 1:
    LogComplicationsProbabilityAnswer = 'не очікується'
  else:
    LogComplicationsProbabilityAnswer = 'очікується'
  LogComplicationsProbabilityPercent = model.predict_proba(NewPatient)
  LogComplicationsProbabilityPercent = LogComplicationsProbabilityPercent[-1][1]
  LogComplicationsProbabilityPercent = LogComplicationsProbabilityPercent*100

  return LogComplicationsProbabilityAnswer, LogComplicationsProbabilityPercent

# де 0 означає відсутність ускладнень, а 1 - наявність

# Дані
Treatment_type_Dic = {
    1: 'Класична_APC - 30 ВТ. 0.8 л_хв',
    2: 'Hibrid NaCl 0.9_Індігокарміну 0.2',
    3: 'Hibrid Гелоспан 4_Індігокарміну 0.2'
}
df = pd.read_csv('BarrettTreatmentComplication.csv')

df['RefluxEsophagitis'] = df['RefluxEsophagitis'].map(RefluxEsophagitisTypes)
df['Effective'] = np.where((df['Complication'] == 0), 1, 0)

features = ['Age', 'RefluxEsophagitis', 'TypeofHernia', 'Type2Diabetes', 'Obesity',
            'Prague_Сlassification_M', 'Level_GSOD_M']

# Тренуємо окрему модель для кожного типу лікування
models = {}
effectiveness_scores = {}
for treatment_id, treatment_name in Treatment_type_Dic.items():
    treatment_data = df[df['Endoscopic_inter'] == treatment_id]
    X = treatment_data[features]
    y = treatment_data['Effective']
    if len(y.unique()) < 2:
      
        continue

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=70, random_state=42)
    model.fit(X_train, y_train)
    models[treatment_id] = model
#========

def recommend_best_treatment(patient_data: dict):
    effectiveness_results = {}
    for treatment_id, model in models.items():
        input_df = pd.DataFrame([patient_data])
        predicted_proba = model.predict_proba(input_df)[0][1]  # Імовірність ефективності
        treatment_name = Treatment_type_Dic[treatment_id]
        effectiveness_results[treatment_name] = predicted_proba
    # Вибір найефективнішого
    best_treatment = max(effectiveness_results, key=effectiveness_results.get)

    effectiveness_results_str_dic = {key: str(value) for key, value in effectiveness_results.items()}
    #return f"\n✅ Рекомендоване лікування: {best_treatment} (найвища ефективність)"
    return best_treatment, effectiveness_results_str_dic

def permute_feature(df, feature):

    permuted_df = df.copy(deep=True) 
    permuted_features = np.random.permutation(permuted_df[feature])
    permuted_df[feature] = permuted_features
    return permuted_df

def permutation_importance():
  num_samples = 100
  df = pd.read_csv("BarrettTreatmentComplication.csv")
  df['RefluxEsophagitis']= df['RefluxEsophagitis'].map(RefluxEsophagitisTypes)
  XColumns = ['Age','Sex', 'RefluxEsophagitis', 'Type2Diabetes', 'Prague_Сlassification_M', 'Level_GSOD_M']
  X_Treatment = df.loc[:, XColumns]
  y_Treatment = df.Complication
  lr = linear_model.LogisticRegression()
  lr.fit(X_Treatment, y_Treatment)

  importances = pd.DataFrame(index = ['importance'], columns = X_Treatment.columns)

  baseline_performance = concordance_index(y_Treatment, lr.predict_proba(X_Treatment)[:, 1])

  for feature in importances.columns:
    feature_performance_arr = np.zeros(num_samples)
    for i in range(num_samples):
      perm_X = permute_feature(X_Treatment,feature)
      feature_performance_arr[i] = concordance_index(y_Treatment, lr.predict_proba(perm_X)[:, 1])
    importances[feature]['importance'] = baseline_performance - np.mean(feature_performance_arr)
  importances_dict = importances.iloc[0].to_dict()
  importances_dict_sorted = heapq.nlargest(6, importances_dict.items(), key=lambda item: item[1])

  return importances_dict_sorted
importances_dict_sorted = permutation_importance()

bot = telebot.TeleBot(os.getenv("BOT_TOKEN"))
#bot = telebot.TeleBot('8017031200:AAEITscCqPkPbpgfqmIasR9PHNqXlXVNRo0')
#Here is the token for bot HiatusHerniaBarrettsDesease @HiatusHerniaBarrettsDesease_bot:

@bot.message_handler(commands=['help', 'start'])

def send_welcome(message):
    msg = bot.reply_to(message, """Привіт я бот \"Barrett Treatment Effect Estimation\"! \n\nЯ допоможу спрогнозувати ефективність лікування \n\nБудь ласка, введіть ім'я пацієнта """)
    chat_id = message.chat.id
    bot.register_next_step_handler(msg, process_name_step)

def process_name_step(message):
    try:
        chat_id = message.chat.id
        name = message.text
        Barrettpatient_dict['name'] = name
        msg = bot.reply_to(message, 'Введіть вік пацієнта')
        bot.register_next_step_handler(msg, process_age_step)
    except Exception as e:
        bot.reply_to(message, 'oooops')

def process_age_step(message):
  try:
    chat_id = message.chat.id
    age_message = message.text
    if not age_message.isdigit():
      msg = bot.reply_to(message, 'Вік має бути цифрою. Будь ласка, введіть вік пацієнта')
      bot.register_next_step_handler(msg, process_age_step)
    else:
      global age
      age = int(age_message)
      Barrettpatient_dict['age'] = int(age_message)
      markup = types.ReplyKeyboardMarkup(one_time_keyboard=False)
      markup.add(g_1, g_0)
      msg = bot.reply_to(message, 'Яка стать?', reply_markup=markup)
      bot.register_next_step_handler(msg, process_gender_step)
  except Exception as e:
    bot.reply_to(message, 'oooops')

def process_gender_step(message):
  try:
    chat_id = message.chat.id
    gender_message = message.text
    if (gender_message == g_1) or (gender_message == g_0):
      Barrettpatient_dict['gender'] = Gender[gender_message]

      markup = types.ReplyKeyboardMarkup(one_time_keyboard=False)
      markup.add(RF_0, RF_1, RF_2, RF_3, RF_4)
      msg = bot.reply_to(message, 'Наявність Рефлюкс-езофагіт?', reply_markup=markup)
      bot.register_next_step_handler(msg, process_RefluxEsophagitis_step)
    else:
      raise Exception("Рефлюкс-езофагіт невідома")
  except Exception as e:
    bot.reply_to(message, 'oooops')

def process_RefluxEsophagitis_step(message):
  try:
    chat_id = message.chat.id
    RefluxEsophagitis_message = message.text

    if (RefluxEsophagitis_message == RF_0) or (RefluxEsophagitis_message == RF_1) or (RefluxEsophagitis_message == RF_2) or (RefluxEsophagitis_message == RF_3) or (RefluxEsophagitis_message == RF_4):
      Barrettpatient_dict['RefluxEsophagitis'] = RefluxEsophagitisTypes[RefluxEsophagitis_message]
      global RefluxEsophagitis
      RefluxEsophagitis = int(RefluxEsophagitisTypes[RefluxEsophagitis_message])
      markup_remove = types.ReplyKeyboardRemove(selective=False)
      msg = bot.reply_to(message, 'Рівень ГСОД?', reply_markup=markup_remove)
      bot.register_next_step_handler(msg, process_Level_GSOD_M_step)
    else:
      raise Exception("RefluxEsophagitis невідома")
  except Exception as e:
    bot.reply_to(message, 'oooops RefluxEsophagitis')

def process_Level_GSOD_M_step(message):
  try:
    chat_id = message.chat.id
    Level_GSOD_M_message = message.text
    if not Level_GSOD_M_message.isdigit():
      msg = bot.reply_to(message, 'Рівень ГСОД має бути цифрою.')
      bot.register_next_step_handler(msg, process_Level_GSOD_M_step)
    else:
      global Level_GSOD_M
      Level_GSOD_M = int(Level_GSOD_M_message)
      
      markup = types.ReplyKeyboardMarkup(one_time_keyboard=False)
      markup.add('1', '2', '3', '4')
      msg = bot.reply_to(message, 'Тип грижі?', reply_markup=markup)
      bot.register_next_step_handler(msg, process_TypeOfHernia_step)
  except Exception as e:
    bot.reply_to(message, 'oooops process_Level_GSOD_M_step')

def process_TypeOfHernia_step(message):
  try:
    chat_id = message.chat.id
    TypeOfHernia_message = message.text
    if (TypeOfHernia_message == '1') or (TypeOfHernia_message == '2') or (TypeOfHernia_message == '3') or (TypeOfHernia_message == '4'):
      global TypeOfHernia
      TypeOfHernia = int(TypeOfHernia_message)
      markup = types.ReplyKeyboardMarkup(one_time_keyboard=False)
      markup.add(o_0, o_1)
      msg = bot.reply_to(message, 'Наявність ожиріння?', reply_markup=markup)
      bot.register_next_step_handler(msg, process_Obesity_step)
    else:
      raise Exception("Невідома process_TypeOfHernia_step")
  except Exception as e:
    bot.reply_to(message, 'oooops process_TypeOfHernia_step')

def process_Obesity_step(message):
  try:
    chat_id = message.chat.id
    Obesity_message = message.text
    if (Obesity_message == o_0) or (Obesity_message == o_1):
      Barrettpatient_dict['Obesity'] = Obesity_dic[Obesity_message]
      global Obesity
      Obesity = int(Obesity_dic[Obesity_message])
      markup_remove = types.ReplyKeyboardRemove(selective=False)
      msg = bot.reply_to(message, 'Пражська класифікація?', reply_markup=markup_remove)
      bot.register_next_step_handler(msg, process_Prague_Сlassification_M_step)
    else:
      raise Exception("Цукрового діабету 2 типу невідома")
  except Exception as e:
    bot.reply_to(message, 'oooops Obesity')

def process_Prague_Сlassification_M_step(message):
  try:
    chat_id = message.chat.id
    Prague_Сlassification_M_message = message.text
    if not Prague_Сlassification_M_message.isdigit():
      msg = bot.reply_to(message, 'Пражська класифікація має бути цифрою.')
      bot.register_next_step_handler(msg, process_Prague_Сlassification_M_step)
    else:
      global Prague_Сlassification_M
      Prague_Сlassification_M = int(Prague_Сlassification_M_message)
      markup = types.ReplyKeyboardMarkup(one_time_keyboard=False)
      markup.add(d_1, d_0)
      msg = bot.reply_to(message, 'Наявність цукрового діабету 2 типу?', reply_markup=markup)
      
      bot.register_next_step_handler(msg, process_Type2Diabetes_step)
  except Exception as e:
    bot.reply_to(message, 'oooops process_Prague_Сlassification_M_step')

def process_Type2Diabetes_step(message):
  try:
    chat_id = message.chat.id
    Type2Diabetes_message = message.text
    if (Type2Diabetes_message == d_1) or (Type2Diabetes_message == d_0):
      Barrettpatient_dict['Type2Diabetes'] = Type2Diabetes_dic[Type2Diabetes_message]
      global Type2Diabetes
      Type2Diabetes = int(Type2Diabetes_dic[Type2Diabetes_message])
      markup = types.ReplyKeyboardMarkup(one_time_keyboard=False)
      markup.add(HC_0, HC_1, HC_2, HC_3)
      msg = bot.reply_to(message, 'Гістологічне заключення?', reply_markup=markup)
      bot.register_next_step_handler(msg, process_TypeofHistological_conclusion_step)
    else:
      raise Exception("Гістологічне заключення невідома")
  except Exception as e:
    bot.reply_to(message, 'oooops Type2Diabetes')

def process_TypeofHistological_conclusion_step(message):
  try:
    chat_id = message.chat.id
    TypeofHistological_conclusion = message.text

    if (TypeofHistological_conclusion == HC_0) or (TypeofHistological_conclusion == HC_1) or (TypeofHistological_conclusion == HC_2) or (TypeofHistological_conclusion == HC_3):

      Barrettpatient_dict['TypeofHistological_conclusion'] = Type_of_Histological_conclusion[TypeofHistological_conclusion]
      markup = types.ReplyKeyboardMarkup(one_time_keyboard=False)
      markup.add(EI_0, EI_1, EI_2)
      msg = bot.reply_to(message, 'Ендоскопічне втручання?', reply_markup=markup)

      bot.register_next_step_handler(msg, process_Endoscopic_inter_Complication_step)
    else:
      raise Exception("Ендоскопічне втручання невідома")
  except Exception as e:
    bot.reply_to(message, 'oooops process_TypeofHistological_conclusion_step')

def process_Endoscopic_inter_Complication_step(message):
  try:
    chat_id = message.chat.id
    Endoscopic_inter = message.text

    if (Endoscopic_inter == EI_0) or (Endoscopic_inter == EI_1) or (Endoscopic_inter == EI_2) or (Endoscopic_inter == EI_3) or (Endoscopic_inter == EI_4):
      Barrettpatient_dict['Endoscopic_inter'] = Treatment_type_of_endoscopic_intervention[Endoscopic_inter]
      gender_P = Barrettpatient_dict['gender']
      TypeofHistological_conclusion_P = Barrettpatient_dict['TypeofHistological_conclusion']
      Endoscopic_inter_P = Barrettpatient_dict['Endoscopic_inter']
### Function
      feature_importance_dict, ComplicationsProbabilityAnswer, ComplicationsProbabilityPercent = RandomForestComplicationsProbabilityFunc(age, gender_P, RefluxEsophagitis, TypeOfHernia, Type2Diabetes, Obesity, Prague_Сlassification_M, Level_GSOD_M, TypeofHistological_conclusion_P, Endoscopic_inter_P)
      LogComplicationsProbabilityAnswer, LogComplicationsProbabilityPercent = LogisticRegressionComplicationsProbabilityFunc(age, gender_P, RefluxEsophagitis, TypeOfHernia, Type2Diabetes, Obesity, Prague_Сlassification_M, Level_GSOD_M, TypeofHistological_conclusion_P, Endoscopic_inter_P)
      bot.send_message(chat_id,

      '\n - Ускладнення ' + str(ComplicationsProbabilityAnswer)+
      '\n - Ймовірність ускладнення у відсотках: ' + str(ComplicationsProbabilityPercent) + ' %' +
      '\n'+ '(RandomForest)'

      '\n\n - Ускладнення ' + str(LogComplicationsProbabilityAnswer)+
      '\n - Ймовірність ускладнення у відсотках: ' + str(LogComplicationsProbabilityPercent) + ' %'  +
      '\n' + '(LogisticRegression)' +
      '\n______________________________________' +

      #'\n\n - Survival ' + str(SurvivalProbabilityAnswer)+
      #'\n - Survival probability in percent: ' + str(SurvivalProbabilityPercent) + ' %'
      #'\n' +
      #'______________________________________' +

      #'\n\n - Probability of neurological outcome (significant recovery vs. disability) ' + str(NeurologicalOutcomeProbabilityAnswer)+
      #'\n - Probability of neurological outcome (significant recovery vs. disability) in percent ' + str(NeurologicalOutcomeProbabilityPercent) + ' %'
      #'\n' +

      #'______________________________________' +
      '\n\n - Важливість факторів(Importance of factors)\n' +
      str(feature_importance_dict)
      )

      markup = types.ReplyKeyboardMarkup(one_time_keyboard=False)
      markup.add('Далі')
      msg = bot.reply_to(message, 'Для прогнозування найефективнішого лікування натисніть Далі.', reply_markup=markup)
      bot.register_next_step_handler(msg, process_The_most_effective_treatment_step)

  except Exception as e:
    bot.reply_to(message, 'oooops process_Endoscopic_inter_Complication_step')

def process_Endoscopic_inter_step(message):
  try:
    chat_id = message.chat.id
    Endoscopic_inter = message.text

    if (Endoscopic_inter == EI_0) or (Endoscopic_inter == EI_1) or (Endoscopic_inter == EI_2) or (Endoscopic_inter == EI_3) or (Endoscopic_inter == EI_4):

      Barrettpatient_dict['Endoscopic_inter'] = Treatment_type_of_endoscopic_intervention[Endoscopic_inter]

      age_P = Barrettpatient_dict['age']
      gender_P = Barrettpatient_dict['gender']
      RefluxEsophagitis_P = Barrettpatient_dict['RefluxEsophagitis']
      Type2Diabetes_P = Barrettpatient_dict['Type2Diabetes']
      Obesity_P = Barrettpatient_dict['Obesity']
      TypeofHistological_conclusion_P = Barrettpatient_dict['TypeofHistological_conclusion']
      Endoscopic_inter_P = Barrettpatient_dict['Endoscopic_inter']

      PredictecomplicationProbabilites = lr.predict_proba([[age_P, gender_P, RefluxEsophagitis_P , Type2Diabetes_P, Obesity_P, TypeofHistological_conclusion_P , Endoscopic_inter_P ]])
      BarrettPrediction = PredictecomplicationProbabilites[-1][1]
      BarrettPrediction = BarrettPrediction*100
      Predictecomplication = lr.predict([[age_P, gender_P, RefluxEsophagitis_P , Type2Diabetes_P, Obesity_P, TypeofHistological_conclusion_P , Endoscopic_inter_P ]])
      if Predictecomplication[0] == 1:
        P_Comlication = "\nУскладнення предбачается\n "
      else:
        P_Comlication = "\nУскладнення не предбачается.\n"

      theta_Endoscopic_inter, Endoscopic_inter_OR, RefluxEsophagitis_OR, coeffs = Endoscopic_intervention_effect(lr, X)
      Endoscopic_inter_OR_Complication = (1 - Endoscopic_inter_OR)
      coeffs = sorted(coeffs.items(), key=lambda x:x[1],  reverse=True)
      coeffs_0 = coeffs[0]
      coeffs_0_0 = coeffs_0[0]
      coeffs_0_1 = coeffs_0[1]
      coeffs_1 = coeffs[1]
      coeffs_1_0 = coeffs_1[0]
      coeffs_1_1 = coeffs_1[1]
      coeffs_2 = coeffs[2]
      coeffs_2_0 = coeffs_2[0]
      coeffs_2_1 = coeffs_2[1]
      coeffs_3 = coeffs[3]
      coeffs_3_0 = coeffs_3[0]
      coeffs_3_1 = coeffs_3[1]
      coeffs_4 = coeffs[4]
      coeffs_4_0 = coeffs_4[0]
      coeffs_4_1 = coeffs_4[1]
      coeffs_5 = coeffs[5]
      coeffs_5_0 = coeffs_5[0]
      coeffs_5_1 = coeffs_5[1]
      coeffs_6 = coeffs[6]
      coeffs_6_0 = coeffs_6[0]
      coeffs_6_1 = coeffs_6[1]

      bot.send_message(chat_id,
                    '\n\nЕфекта від Лікування: ' + Endoscopic_inter + ' на процес ускладнення захворювання: ' + '\n' + str(theta_Endoscopic_inter)
                    + '\nЛікування зменшує співвідношення шансів ускладнення ' + str(Endoscopic_inter_OR) + ' разів.'
                    + '\n\nИмовірність ускладнення при застосуванні: '+ Endoscopic_inter + '\n' + str(BarrettPrediction) + ' %' +
                    '\n' + P_Comlication
                    + '-----------------------\n'

                    + '\n\nІнші коефіцієнти: \n'
                    + str(coeffs_0_0) +': ' + str(coeffs_0_1) + '\n'
                    + str(coeffs_1_0) +': ' + str(coeffs_1_1) + '\n'
                    + str(coeffs_2_0) +': ' + str(coeffs_2_1) + '\n'
                    + str(coeffs_3_0) +': ' + str(coeffs_3_1) + '\n'
                    + str(coeffs_4_0) +': ' + str(coeffs_4_1) + '\n'
                    + str(coeffs_5_0) +': ' + str(coeffs_5_1) + '\n'
                    + str(coeffs_6_0) +': ' + str(coeffs_6_1) + '\n'
                    + '\nНегативне значення коефіцієнта  вказує на те, що вибраний параметр знижує ймовірність розвитку ускладнень.'
                       )

      markup = types.ReplyKeyboardMarkup(one_time_keyboard=False)
      markup.add('Далі')
      msg = bot.reply_to(message, 'Для прогнозування найефективнішого лікування натисніть Далі.', reply_markup=markup)
      bot.register_next_step_handler(msg, process_The_most_effective_treatment_step)
    else:
      raise Exception("Блок текта")
  except Exception as e:
    bot.reply_to(message, 'oooops Блок текта')
#################################################

def process_The_most_effective_treatment_step(message):
  try:
    chat_id = message.chat.id
    The_most_effective_treatment_message = message.text

    if (The_most_effective_treatment_message == 'Далі'):
      new_patient = {'Age': age, 'RefluxEsophagitis': RefluxEsophagitis, 'TypeofHernia': TypeOfHernia, 'Type2Diabetes': Type2Diabetes, 'Obesity': Obesity, 'Prague_Сlassification_M': Prague_Сlassification_M, 'Level_GSOD_M': Level_GSOD_M}
      best_treatment, effectiveness_results_str_dic = recommend_best_treatment(new_patient)
      Classic = effectiveness_results_str_dic['Класична_APC - 30 ВТ. 0.8 л_хв']
      Hibrid_NaCal = effectiveness_results_str_dic['Hibrid NaCl 0.9_Індігокарміну 0.2']
      Hibrid_Gelospan = effectiveness_results_str_dic['Hibrid Гелоспан 4_Індігокарміну 0.2']
      bot.send_message(chat_id,
      '\n\nРекомендоване лікування: \n' + str(best_treatment) + ' (найвища ефективність)' +
      '\n\nПрогнозована ефективність для кожного виду лікування: ' +
      '\n - Класична_APC - 30 ВТ. 0.8 л_хв:  '  + str(Classic) + ' %'+
      '\n - Hibrid NaCl 0.9_Індігокарміну 0.2:  '  + str(Hibrid_Gelospan) + ' %'+
      '\n - Hibrid Гелоспан 4_Індігокарміну 0.2:  '  + str(Hibrid_Gelospan) + ' %' +
      '\n\n' +
      ''' '''
      )

      markup = types.ReplyKeyboardMarkup(one_time_keyboard=False)
      markup.add('Permutation')
      msg = bot.reply_to(message, 'Виберіть дію.', reply_markup=markup)
      bot.register_next_step_handler(msg, process_Permutation_step)
    else:
      raise Exception("process_Recommendations_step")
  except Exception as e:
    bot.reply_to(message, 'oooops process_The_most_effective_treatment_step')

def process_Permutation_step(message):
  try:
    chat_id = message.chat.id
    Permutation_answer = message.text
    if (Permutation_answer == 'Permutation'):
      importances_key_1, importances_value_1 = importances_dict_sorted[0]
      importances_key_2, importances_value_2 = importances_dict_sorted[1]
      importances_key_3, importances_value_3 = importances_dict_sorted[2]
      importances_key_4, importances_value_4 = importances_dict_sorted[3]
      importances_key_5, importances_value_5 = importances_dict_sorted[4]
      importances_key_6, importances_value_6 = importances_dict_sorted[5]
      bot.send_message(chat_id,
                       '\nВажливості симптомів:  \n'
                       + '\n' + importances_key_1 + ': ' + str("{:.4f}".format(importances_value_1))
                       + '\n' + importances_key_2 + ': ' + str("{:.4f}".format(importances_value_2))
                       + '\n' + importances_key_3 + ': ' + str("{:.4f}".format(importances_value_3))
                       + '\n' + importances_key_4 + ': ' + str("{:.4f}".format(importances_value_4))
                       + '\n' + importances_key_5 + ': ' + str("{:.4f}".format(importances_value_5))
                       + '\n' + importances_key_6 + ': ' + str("{:.4f}".format(importances_value_6))
                       )

      markup = types.ReplyKeyboardMarkup(one_time_keyboard=False)
      markup.add('Далі')
      msg = bot.reply_to(message, 'Спробувати знову.', reply_markup=markup)
      bot.register_next_step_handler(msg, send_welcome)
    #else:
    #  raise Exception("process_Recommendations_step")
  except Exception as e:
    bot.reply_to(message, 'oooops process_Permutation_step')

bot.infinity_polling()

