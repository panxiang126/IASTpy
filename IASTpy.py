#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
Program name:
IASTpy
Description:

Author:
Mr. Pan Xiang
Email:
panxiang126@gmail.com
Dependency:
This program uses pandas, scipy, numpy, matplotlib. You can install them by one command: pip install pandas scipy numpy matplotlib pyinstaller

# pyinstaller IASTpy.py --onefile -i logo.ico

'''
import pandas as pd
import scipy as sp
import numpy as np
from scipy.optimize import curve_fit
from scipy.optimize import fminbound
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib.pyplot import MultipleLocator
#import logging
import time

time_line = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
#logging.basicConfig(filename=time_line+"_log.txt", level=logging.DEBUG, format="%(message)s")


def printOut(po):
  with open(time_line+"_log.txt", "a+") as fw:
    print(po, file=fw)
  print(po)

  
printOut("""
IAST implemented in python
Author: Pan Xiang
Any questions and requests, please contact the author by panxiang126@gmail.com
""")


print("""
IAST implemented in python
Author: Pan Xiang
Any questions and requests, please contact the author by panxiang126@gmail.com
""")

def quitScript(pot):
  import sys
  print(pot)
  input('press any key to quit script...')
  sys.exit()

readDir = input("please enter the data directory.\n(defult: "+r"molData.csv): ")
readDir = "molData.csv" if readDir == "" else readDir.replace('"','')

try:
  if "csv" in readDir:
    df_mol = pd.read_csv(readDir, header=None, engine='python')#, sep=',')
  #elif "xlsx" in readDir:
  #  df_mol = pd.read_excel(readDir, header=None, engine='python')#, sep=',')
except FileNotFoundError as error:
  print(error)
  quitScript()
else:
  print(f"read file from {readDir}:")
  print(df_mol, "\n\n")

df_mol = df_mol.replace("--",np.nan)
df_mol = df_mol.dropna(axis=0, how="all")
df_mol = df_mol.dropna(axis=1, how="all")
df_mol = df_mol.astype("float64")

pre_unit_list=[1.0, 1/1000, 1/100, 1.01325]
pre_unit = input("please enter the unit of pressure. 1 for bar, 2 for mbar, 3 for kPa, or 4 for atm. \nfor other unit, please covert one of above, and restart the script.\n(defult: 1): ")
if pre_unit == "":
  pre_unit = 1
elif eval(pre_unit) in range(1,len(pre_unit_list)+1):
  pre_unit=eval(pre_unit)
else:
  quitScript("invalid input!")

ad_unit_list=[1/22.4, 1.0]
ad_unit = input("Please enter the unit of gas adsorption. 1 for cm-3/g or 2 for mmol/g. \nfor other unit, please covert one of above, and restart the script.\n(defult: 1): ")
if ad_unit == "":
  ad_unit = 1
elif eval(ad_unit) in range(1,len(ad_unit_list)+1):
  ad_unit=eval(ad_unit)
else:
  quitScript("invalid input!")

df_mol.columns = ["a_Pressure", "a_Loading", "b_Pressure", "b_Loading"]

df_mol_A=df_mol[["a_Pressure", "a_Loading"]]
df_mol_A = df_mol_A.dropna(axis=0, how="all")
df_mol_A.loc[:]["a_Pressure"] *= pre_unit_list[pre_unit-1]
df_mol_A.loc[:]["a_Loading"] *= ad_unit_list[ad_unit-1]

df_mol_B=df_mol[["b_Pressure", "b_Loading"]]
df_mol_B = df_mol_B.dropna(axis=0, how="all")
df_mol_B.loc[:]["b_Pressure"] *= pre_unit_list[pre_unit-1]
df_mol_B.loc[:]["b_Loading"] *= ad_unit_list[ad_unit-1]

pd.set_option('display.max_rows', 6)

printOut("\n\npart 1 read as:")
printOut(df_mol_A)
printOut("\n")
printOut("part 2 read as:")
printOut(df_mol_B)
printOut("\n\nplease check the input data carefully, where the units of Pressure and Loading have been converted to bar and mmol/g.\n")
input("\npress the Enter key to continue...")

print("\n\n\nfitting using Langmuir model...")
def func_s(p, a, b, c):
  return a * b * (p**c)/(1 + b * (p ** c))
def func_d(p, a, b, c, d, e, f):
  return a * b * (p**c)/(1 + b * (p ** c)) + d * e * (p**f)/(1 + e * (p ** f))

fit_md = input("please select the fitting formula: 1 for single site, 2 for double site.\n(defult: 1): ")
if fit_md == "" or eval(fit_md) == 1:
  func = func_s
elif eval(fit_md) == 2:
  func = func_d
else:
  quitScript("invalid input!")

def plotFit(saveName, popt, xdata, ydata, r2a):
  title_size = 32
  fig_size =[10*0.8, 7.65*0.8]
  plt.figure(figsize=fig_size)
  plt.rc("font",family="Arial")

  ax_th = 2
  x_tick_size = 24
  y_tick_size = 24
  title_size = 32
  x_label_size = 26
  y_label_size = 26
  legend_size = 18
  x_label_content = "Pressure (bar)"
  y_label_content = "Gas adsorption (mmol/g)"
  x_max = max(xdata)
  x_min = min(xdata)
  y_max = max(ydata)
  y_min = min(ydata)

  ax = plt.gca()
  ax.spines["bottom"].set_linewidth(ax_th)
  ax.spines["left"].set_linewidth(ax_th)
  ax.spines["top"].set_linewidth(ax_th)
  ax.spines["right"].set_linewidth(ax_th)
  ax.xaxis.set_minor_locator(AutoMinorLocator(2))
  ax.yaxis.set_minor_locator(AutoMinorLocator(2))
  ax.axes.tick_params(direction="out", length=8, width=ax_th, which="major")
  ax.axes.tick_params(direction="out", length=4, width=ax_th, which="minor")
  plt.xlim(x_min-0.10*(x_max-x_min),x_max+0.10*(x_max-x_min))
  plt.ylim(y_min-0.10*(y_max-y_min),y_max+0.10*(y_max-y_min))
  plt.xticks(font={"size":x_tick_size})
  plt.yticks(font={"size":y_tick_size})
  plt.xlabel(x_label_content,font={"size":x_label_size})
  plt.ylabel(y_label_content,font={"size":y_label_size})

  plt.scatter(xdata, ydata, s=160, c="r", marker="^", label="Adsorption point", zorder=0)
  x = np.linspace(x_min, x_max,100)
  plt.plot(x,func(x, *popt), color='b',linewidth=4.0,linestyle='-', label="Fitted curve", zorder=1)

  if func == func_s:
    plt.text(0.7*(x_max-x_min), 0.4*(y_max-y_min), r"$n^{\circ}(p)="+f"{popt[0]:.3f}"+r"\times \frac{"+f"{popt[1]:.3f}"+r"\times p^{"+f"{popt[2]:.3f}"+"}}{1+"+f"{popt[1]:.3f}"+r"\times p^{"+f"{popt[2]:.3f}"+"}}$"+"\nR$^2$: "+f"{r2a:.3f}", fontsize=0.8*legend_size,rotation=0, horizontalalignment = "center", verticalalignment = "top", bbox={"fc":"1", "ec":"None", "alpha":0.8, "pad":0})

  plt.legend(loc="upper left", prop={"size":legend_size}, # bbox_to_anchor=(1.45, 1.0),
          markerscale=1.0,scatteryoffsets=[0.5], handletextpad=0.5, ncol=1, labelspacing=0.8,
          columnspacing=0.2, frameon=True, framealpha=0.9, edgecolor="w")

  # save.
  saveDPI = 300
  #plt.savefig(f"{saveName}.tif", dpi=saveDPI, bbox_inches="tight")
  #plt.savefig(f"{saveName}.jpg", dpi=saveDPI, bbox_inches="tight")
  plt.savefig(f"{saveName}.svg", format="svg", bbox_inches="tight")
  #plt.show()


def fit_data(fit_name, func, xdata, ydata):
  print(f"\nfitting {fit_name}...")
  fit_method = ["trf","dogbox","lm"]
  fit_param =[]
  fit_r2 = [] 

  fail_n = 0
  for md in fit_method:
    print(f"method of {md}:", end=" ")
    try:
      popt, pcov = curve_fit(func, xdata, ydata, maxfev=10000, method=md, bounds=(-np.inf, np.inf))
    except:
      fail_n += 1
      print("fit failed")
      fit_param.append("fit_fail")
      fit_r2.append(-1E20)
    else:
      ave_ydata = np.mean(ydata)
      suma_up = 0
      suma_dn  = 0
      for i,j in zip(ydata, func(xdata, *popt)):
        suma_up  += (j-i)**2
        suma_dn  += (j-ave_ydata)**2
      r2a = 1-(suma_up)/(suma_dn)
      print(popt, " R2:", r2a)
      fit_param.append(popt)
      fit_r2.append(r2a)

  if fail_n == len(fit_method):
    quitScript("all fit failed, please check raw data.")
  else:
    best_n = fit_r2.index(max(fit_r2))
  
  printOut(f"use method of {fit_method[best_n]} {fit_param[best_n]} R2: {fit_r2[best_n]}")
  plotFit(fit_name, fit_param[best_n], xdata, ydata, fit_r2[best_n])
  return fit_method[best_n],fit_param[best_n],fit_r2[best_n]

md_1, fp_1, r2a_1 = fit_data("part_1", func, df_mol_A.loc[:]["a_Pressure"], df_mol_A.loc[:]["a_Loading"])
md_2, fp_2, r2a_2 = fit_data("part_2", func, df_mol_B.loc[:]["b_Pressure"], df_mol_B.loc[:]["b_Loading"])


def plotSP(saveNme, pt, sp):
  import matplotlib.pyplot as plt
  from matplotlib.ticker import AutoMinorLocator
  from matplotlib.pyplot import MultipleLocator

  title_size = 32
  fig_size =[10*0.8, 7.65*0.8]
  plt.figure(figsize=fig_size)
  plt.rc("font",family="DejaVu Sans")

  ax_th = 2
  x_tick_size = 24
  y_tick_size = 24
  title_size = 32
  x_label_size = 26
  y_label_size = 26
  legend_size = 18
  x_label_content = "Pressure (bar)" 
  y_label_content = "Separation factor"
  x_max = max(pt)
  x_min = min(pt)
  y_max = max(sp)
  y_min = min(sp)

  ax = plt.gca()
  ax.spines["bottom"].set_linewidth(ax_th)
  ax.spines["left"].set_linewidth(ax_th)
  ax.spines["top"].set_linewidth(ax_th)
  ax.spines["right"].set_linewidth(ax_th)
  ax.xaxis.set_minor_locator(AutoMinorLocator(2))
  ax.yaxis.set_minor_locator(AutoMinorLocator(2))
  ax.axes.tick_params(direction="out", length=8, width=ax_th, which="major")
  ax.axes.tick_params(direction="out", length=4, width=ax_th, which="minor")
  plt.xlim(x_min-0.10*(x_max-x_min),x_max+0.10*(x_max-x_min))
  plt.xticks(font={"size":x_tick_size})
  plt.yticks(font={"size":y_tick_size})
  plt.xlabel(x_label_content,font={"size":x_label_size})
  plt.ylabel(y_label_content,font={"size":y_label_size})
  
  if max(sp)/min(sp) > 1E5 or max(sp)/min(sp) < 1E-5:
    ax.set_yscale('log')
  else:
    plt.ylim(y_min-0.10*(y_max-y_min),y_max+0.10*(y_max-y_min))

  plt.plot(pt, sp, c="b", linewidth=3, linestyle="-", zorder=0)

  # save.
  saveDPI = 300
  #plt.savefig(f"{saveNme}.tif", dpi=saveDPI, bbox_inches="tight")
  #plt.savefig(f"{saveNme}.jpg", dpi=saveDPI, bbox_inches="tight")
  plt.savefig(f"{saveNme}.svg", format="svg", bbox_inches="tight")
  #plt.show()


print("\n\n\ncalculation separation factor...")

p_list_in = input("please enter the total pressure (in bar).\n(defult: [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 20.0, 24.0, 28.0]): ")

if p_list_in == "":
  p_list = [ (i+1)/100 for i in range(10) ]+[ (i+1)/20 for i in range(2, 20) ]+[2, 4, 8, 12, 16, 20, 24, 28]
elif isinstance(eval(p_list_in), list):
  p_list = eval(p_list_in)
elif isinstance(eval(p_list_in), float):
  p_list = [eval(p_list_in)]
else:
  quitScript("invalid input!")


while True:
  y_1 = input("Please enter the content of component part_1, and the value of part_2 will be set as 1-part_1; q to quit.\n(0 < part_1 < 1, defult: 0.5): ")
  if y_1 == "":
    y_1 = 0.5 
  elif y_1 == "q":
    break
  elif 0<eval(y_1)<1:
    y_1 = eval(y_1)
  else:
    quitScript("invalid input!")

  y_2 = 1 - y_1

  df=pd.DataFrame(columns=("Pt (bar)", "x1", "x2", "P10 (bar)","P20 (bar)","error","S12","S21"))
  s12_list=[]
  s21_list=[]


  def f_diff(x_1):
    x_2 = 1 - x_1
    p_1 = y_1*p_target/x_1
    p_2 = y_2*p_target/x_2
    return np.abs(fp_1[0]*np.log(1+fp_1[1]*(p_1**fp_1[2]))/fp_1[2]-fp_2[0]*np.log(1+fp_2[1]*(p_2**fp_2[2]))/fp_2[2])

    
  for p_target in p_list:
    all_list = {}

    #minimum = fminbound(f_diff, 0, 1, xtol=1e-10, maxfun=100000)  
    #final_x_1 = minimum
    #final_error = f_diff(final_x_1)
    #print(final_x_1, final_error)

    #for x_1 in range(9999):
    #  all_list[(x_1+1)/10000]=f_diff((x_1+1)/10000)

    #final_x_1 = min(all_list, key=all_list.get)
    #final_error = min(all_list.values())
    #print(final_x_1, final_error)

  
    try:
      diff_list=[]
      for i in range(1, 10):
        diff_list.append(f_diff(i/10))
      diff_list.index(min(diff_list))+1
      dn = (diff_list.index(min(diff_list))+1)/10
      for it in range(1,20):
        diff_list=[]
        for i in range(-9, 10):
          diff_list.append(f_diff(dn+i/(10**(it+1))))
        dn += (diff_list.index(min(diff_list))+1-10)/(10**(it+1))
      final_x_1 = dn
      final_error = f_diff(final_x_1)
    except ZeroDivisionError:
      print(f"{p_target}: x1 or x2 too small, drop this data")
      p_list.remove(p_target)
      continue


    s12_list.append(final_x_1*y_2/(1-final_x_1)/y_1)
    s21_list.append(1/s12_list[-1])
  
    df.loc[len(df)]=[p_target, final_x_1,1-final_x_1 ,p_target*y_1/final_x_1, p_target*y_2/(1-final_x_1), final_error, s12_list[-1], s21_list[-1]]

  pd.set_option('display.max_rows', None)
  pd.set_option('display.max_columns', None)
  printOut(f"\ncontent of component part_1: {y_1}")
  #printOut(df)
  printOut(df[["Pt (bar)", "S12", "S21"]])
  printOut("\n"*2)
  df.to_csv("sum_p"+str(y_1)+".csv")
  #printOut(f'\np_{y_1} {df[["Pt (bar)", "S12", "S21"]]}')

  plotSP("SP_12_p"+str(y_1), df["Pt (bar)"], df["S12"])
  plotSP("SP_21_p"+str(y_1), df["Pt (bar)"], df["S21"])

input("\n\nall calculations have been done! check the calculation results carefully.\npress the Enter key to exit the script.")
