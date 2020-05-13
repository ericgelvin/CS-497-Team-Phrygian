import os
import pickle
import json

# '{"manner_of_death":0,"armed":0,"age":35,"gender":0,"race":1,"city":2,
#"state":2,"signs_of_mental_illness":1,"threat_level":1,"flee":0,"body_camera":1}'
file = open('./features.dat', 'rb')
features = pickle.load(file)
file.close

command = "curl -i -H \'Content-Type: application/json\" -X POST -d \""


testDat = {}
for feature in features:
	testDat[feature] = 0

testDat['shot'] = 1
testDat['vehicle'] = 1
testDat['M'] = 1
testDat['W'] = 1
testDat['Los Angeles'] = 1
testDat['CA'] = 1
testDat['False'] = 1
testDat['attack'] = 1
testDat['Not fleeing'] = 1
testDat['False'] =1


jsonData = json.dumps(testDat)
command = command + jsonData + "\' http://localhost:5000/predict/sample"

print(command)

os.system(command)