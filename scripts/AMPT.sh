#!/usr/bin/env bash

seed=4321


# dataset
dataDir='clinc'

sourceDomain="travel,work,utility,banking"

valDomain="home,auto_commute,small_talk" 

method="AMPT"

targetDomain="home,auto_commute,small_talk"

testDomain="meta"
testDomain1="kitchen_dining"
testDomain2="credit_cards"

splitargetDomainone="home"

splitargetDomaintwo="auto_commute"

disableCuda=

# training
saveName=bertforampt
learningRate=1e-5
learningRate=5e-6

# model setting
# common
LMName=bert-base-uncased

echo "Start Experiment ..."
logFolder=./log/
mkdir -p ${logFolder}
logFile=${logFolder}/transfer_${sourceDomain}_to_${targetDomain}.log
# logFile=${logFolder}/transfer_mlm_${sourceDomainName}_to_${targetDomainName}_${way}way_${shot}shot.log
if $debug
then
	logFlie=${logFolder}/logDebug.log
fi
for s in 1 2 3 4 5 6 7 8 9 10
    do
	for seed in 1 2 3 4 
		do
		python AMPT.py \
			${disableCuda}\
			--seed ${seed} \
			--valDomain  ${valDomain}  \
				--sourceDomain ${sourceDomain} \
				--targetDomain ${targetDomain} \
				--splitargetDomainone ${splitargetDomainone} \
				--splitargetDomaintwo ${splitargetDomaintwo} \
				--testDomain ${testDomain} \
				--testDomain1 ${testDomain1} \
				--testDomain2 ${testDomain2} \
				--method ${method} \
				--dataDir ${dataDir} \
				--saveModel \
				--learningRate  ${learningRate} \
				--LMName ${LMName} \
				--saveName ${saveName} \
				| tee "${logFile}"
		#   	--tensorboard \
		#   	--mlm \
		echo "Experiment finished."
		done
	done
