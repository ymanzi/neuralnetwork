import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def draw_plot(list_values):
	training = list_values[0]
	test = list_values[1]
	indice = [x for x,y in enumerate(list_values[0])]
	plt.plot(indice, training)#, label="Training")
	plt.plot(indice, test)#, label="Test")
	# plt.xlabel("Normalized KM")
	# plt.ylabel("Price")
	# title = "Cost : " + str(cost)[:9]
	# plt.legend()
	plt.show()



class Komparator:
	def __init__(self, df: pd.DataFrame):
		self.data = df

	def compare_box_plots(self, categorical_var, numerical_var):
		lst_categorical = self.data[categorical_var].unique()
		row = int(len(lst_categorical))
		fig, axs = plt.subplots(nrows= 1, ncols= row)
		for i, elem in enumerate(lst_categorical):
			df = self.data[(self.data[categorical_var] == elem)][numerical_var].dropna()
			axs[i].boxplot(df, vert=False, notch = True, labels=list(lst_categorical[i]), whis=0.75, widths = 0.1, patch_artist=bool(i % 2))
			axs[i].set_xlabel(numerical_var)
			axs[i].legend(lst_categorical[i])
		plt.show()

	def density(self, categorical_var, numerical_var) :
		lst_categorical = self.data[categorical_var].unique()
		for elem in lst_categorical:
			data = self.data[(self.data[categorical_var] == elem)][numerical_var]
			sns.distplot(data, hist=False, kde=True, kde_kws={'linewidth': 3}, label = elem)
		plt.legend(prop={'size': 16}, title=categorical_var)
		plt.show()

	# def pairplot_(self, categorical_var, numerical_var):
	# 	lst_categorical = self.data[categorical_var].unique()
	# 	size_n_var = len(numerical_var) // 2
	# 	for var1 in numerical_var[:size_n_var]:
	# 		for var2 in numerical_var[size_n_var:]:
	# 			# if var1 != var2:
	# 			sns.pairplot(self.data, hue = categorical_var, vars=[var1, var2])
	# 			plt.show()
	# 	return

	def pairplot_(self, categorical_var, numerical_var):
		sns.pairplot(self.data, vars=numerical_var, height=1)
		plt.show()
		return

	def scatterplot_(self, categorical_var, numerical_var):
		size_n_var = len(numerical_var) // 2
		for j, var1 in enumerate(numerical_var):
			tmp_cat = list(numerical_var[j:])
			tmp_cat.remove(var1)
			if len(tmp_cat) > 1:
				if len(tmp_cat) % 2:
					fig, axes = plt.subplots(nrows = 2, ncols = (len(tmp_cat) // 2) + 1, figsize=(15,10))
				else:
					fig, axes = plt.subplots(nrows = 2, ncols = (len(tmp_cat) // 2), figsize=(15,10))
			for i, var2 in enumerate(tmp_cat):
				if len(tmp_cat) == 1:
					sns.scatterplot(data=self.data, y=var1, x=var2,hue = categorical_var)
				elif len(tmp_cat) == 2:
					sns.scatterplot(data=self.data, y=var1, x=var2,hue = categorical_var, ax = axes[i])
				else:
					sns.scatterplot(data=self.data, y=var1, x=var2,hue = categorical_var, ax = axes[i % 2][i // 2])
			plt.show()
		return

	def compare_histograms(self, categorical_var, numerical_var):
		lst_categorical = self.data[categorical_var].unique()
		row = int(len(numerical_var))
		fig, axis = plt.subplots(nrows= 2, ncols=  row // 2 + 1)
		for i in range(0, len(numerical_var)):
			my_list = []
			for j, elem in enumerate(lst_categorical):
				my_list.append (list(self.data[(self.data[categorical_var] == elem)][numerical_var[i]].dropna().values.transpose()))
			axis[i % 2, i // 2].hist(my_list, stacked=False, label=lst_categorical)#, density=True, bins = int(185/15))
			axis[i % 2, i // 2].set_xlabel(numerical_var[i])
			if not(i):
				axis[i % 3, i // 3].legend()
		fig.suptitle("Histograms")
		plt.show()

