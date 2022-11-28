#!/bin/bash

cd ~

python splitExcel.py

myDate=`date +%Y%m%d%H%M%S`
mkdir down_${myDate} 
mv sp*xlsx down_${myDate} 
tar -zvcf down_${myDate}.tgz down_${myDate}
