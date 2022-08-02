import shap
import joblib
import matplotlib.pyplot as plt
import pandas as pd

data=pd.read_excel(r'Experimental.xlsx',sheet_name=0)
model=joblib.load('TrAGBDT.model')
explainer = shap.Explainer(model)
shap_values = explainer(data.iloc[:91,1:-1])

#Explanation for instances
shap.waterfall_plot(shap_values[23],show=False)
shap.waterfall_plot(shap_values[87],show=False)
shap.plots.heatmap(shap_values[:],show=False)

#Feature importance analysis
shap.summary_plot(shap_values,plot_type='bar',show=False)

#PDP analysis
names=data.columns
shap.dependence_plot(names[0], shap_values.values,data.iloc[:91,1:-1],interaction_index=None,show=False)
shap.dependence_plot(names[2], shap_values.values,data.iloc[:91,1:-1],interaction_index=None,show=False)
shap.dependence_plot(names[3], shap_values.values,data.iloc[:91,1:-1],interaction_index=None,show=False)
shap.dependence_plot(names[4], shap_values.values,data.iloc[:91,1:-1],interaction_index=None,show=False)
shap.dependence_plot(names[6], shap_values.values,data.iloc[:91,1:-1],interaction_index=None,show=False)
shap.dependence_plot(names[8], shap_values.values,data.iloc[:91,1:-1],interaction_index=None,show=False)