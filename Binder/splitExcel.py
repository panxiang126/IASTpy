#!/usr/bin/env python
# -*- coding: utf-8 -*-


# pyinstaller splitExcel.py --onefile 


import pandas as pd

readDir = input("please enter the data directory.\n(defult: "+r"split.xlsx): ")
readDir = "split.xlsx" if readDir == "" else readDir.replace('"','')

try:
  if "csv" in readDir:
    df_mol = pd.read_csv(readDir, header=None, engine='python')#, sep=',')
  elif "xlsx" in readDir:
    df_mol = pd.read_excel(readDir, header=None)
except FileNotFoundError as error:
  print(error)
  quitScript()
else:
  print(f"read file from {readDir}:")
  print(df_mol, "\n\n")


spNum = input("please enter the split number.\n(defult: "+r"10) ")
splitNum = 10 if spNum == "" else eval(spNum)


outList=[]
allIndex = [i+1 for i in range(len(df_mol))]
lastIndex = 0
#splitNum=10
for i in allIndex:
  if (i)%splitNum == 0:
    outList.append([lastIndex, i])
    lastIndex = i

if not lastIndex == allIndex[-1]:
  outList.append([lastIndex, allIndex[-1]])

print(outList)

n=0
for i in outList:
  savedf=df_mol[i[0]: i[-1]]
  savedf.to_excel(f"split_{n+1}.xlsx",index=False, header=None)
  n+=1