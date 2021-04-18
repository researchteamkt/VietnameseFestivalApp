from transformers import BertTokenizer
from transformers import BertForQuestionAnswering
from transformers import AutoModel, AutoTokenizer
from transformers import RobertaTokenizer, RobertaForQuestionAnswering
from geopy.geocoders import Nominatim
from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage
import cv2
import os
import torch
import json
from classification_network import img_classification
from owlready2 import *
from flask import Flask, render_template, redirect, url_for, request
app = Flask(__name__)
#====================================
# Festivals
all_festivals = []
id_labels_file="model/index_labels.txt"
#Now read the labels back into a list object
with open(id_labels_file, 'r') as f:
	all_festivals = json.loads(f.read())
	
vn_labels = []
vn_labels_file="model/vn_labels.txt"
#Now read the labels back into a list object
with open(vn_labels_file, 'r') as f:
	vn_labels = json.loads(f.read())

three_festivals = []	

def find_vn_festival(en_name):
	for i in range(len(all_festivals)):
		if all_festivals[i] == en_name:
			return vn_labels[i]
	return ""
		

#Ontology

#onto = get_ontology("C:\Owl\VietnameseFestivalOntology_Ver2.owl")
onto = get_ontology("VFO.owl")
onto.load()
#====================================
#BERT
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
#====================================
def answer_question(question, answer_text):
    input_ids = tokenizer.encode(question, answer_text)
    sep_index = input_ids.index(tokenizer.sep_token_id)

    num_seg_a = sep_index + 1
    num_seg_b = len(input_ids) - num_seg_a
    segment_ids = [0]*num_seg_a + [1]*num_seg_b

    assert len(segment_ids) == len(input_ids)
    # ======== Evaluate ========
    '''
    ResModels = model(torch.tensor([input_ids]),token_type_ids=torch.tensor([segment_ids]))

    # ======== Reconstruct Answer ========
    answer_start = torch.argmax(ResModels.start_logits)
    answer_end = torch.argmax(ResModels.end_logits)
	'''
    start_scores, end_scores = model(torch.tensor([input_ids]), # chuỗi index biểu thị cho inputs.
                                    token_type_ids=torch.tensor([segment_ids])) # chuỗi index thành phần segment câu để phân biệt giữa câu question và câu answer_text

    # ======== Reconstruct Answer ========
    # Tìm ra vị trí start, end với score là cao nhất
    answer_start = torch.argmax(start_scores)
    answer_end = torch.argmax(end_scores)	
#-------------------	
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    answer = tokens[answer_start]

    for i in range(answer_start + 1, answer_end + 1):
        if tokens[i][0:2] == '##':
            answer += tokens[i][2:]
        else:
            answer += ' ' + tokens[i]
    
    print('Question: "' + question + '"')
    print('Answer: "' + answer + '"')
    return answer
   
    
#====================================
#Function Of Website
def QueryFestivals_Place_Time(Place, Month):
    FestivalName = onto.search(is_a = onto[Place],CoThoiGianToChuc = "*/{0}*".format(Month))
    return FestivalName
def QueryFestivals_Description(FestivalName):
    Dict_Place ={}
    Description = list(onto.hasDescription.get_relations())    
    for eachPlace in FestivalName:        
        Dict_Place[eachPlace] = []
        for each in Description:
            if each[0] == eachPlace:                
                Dict_Place[eachPlace] = each[1]
                
    return Dict_Place
def QueryFestivals_OneDescription(FestivalName):
    Dict_Place = ""
    Description = list(onto.hasDescription.get_relations())
    for each in Description:        
        if "{0}".format(each[0]) == FestivalName:                 
            Dict_Place = "{0}".format(each[1])    
    return Dict_Place   
def QueryFestivals_Festival(Festival):
    Festival = onto.search(is_a = onto[Festival])
    return Festival
#=====================================
@app.route("/")
def home():
    return render_template('index.html')


@app.route('/result')
def result():
   FestivalName = QueryFestivals_Place_Time()   
   return render_template('ResultFestival.html', result = FestivalName)
   
@app.route('/Description',methods = ['POST', 'GET'])
def resultDescription():
    Place="SocTrang"
    Month=10
    if request.method == 'POST':
        Place = request.form['Place']        
        Month = request.form['Month']  
    else:
        Place = request.args.get('Place')
        Month = request.args.get('Month')
    FestivalName = QueryFestivals_Place_Time(Place,Month)
    global three_festivals
    print("3 Le Hoi: ", three_festivals)
    test0 = QueryFestivals_Festival(three_festivals[0])
    test1 = QueryFestivals_Festival(three_festivals[1])
    test2 = QueryFestivals_Festival(three_festivals[2])
    #test.is_a.append(QueryFestivals_Festival(three_festivals[1]))
    test = test0
    
    print("Test: ", test)
    print("Le Hoi Onto: ", FestivalName)
    #FestivalName = list(set(FestivalName).intersection(set(three_festivals)))
    FestivalName = list(set(FestivalName).intersection(set(test)))
    print("Le Hoi Onto Sau: ", FestivalName)
    Description = QueryFestivals_Description(FestivalName)   
    print("Ket Qua Mo ta: ", Description)
    return render_template('result.html', Names = FestivalName, Des = Description)

#======================================
#ChatBox
#@app.route("/chatbot")
des=""
@app.route('/chatbot',methods = ['POST', 'GET'])
def chatbot():
    FestivalName = request.args.get('key')
    global des
    des = QueryFestivals_OneDescription(FestivalName)    
    return render_template("chatbot.html", Name = FestivalName)
    
@app.route("/get", methods=["POST"])
def chatbot_response():
    msg = request.form["msg"]
    print(msg)
    question = msg#"Where is this festival?"
    paragraph = des
    res = "Robot: {0}".format(answer_question(question, paragraph))    
    if "[CLS]" in res:
        res = "Sorry, I didn't quite understand what you typed, try asking a more specified question related to festival."
    
    return res
#----------------
@app.route('/Classification',methods = ['POST', 'GET'])
def classification():
    if request.method == 'POST':
        f = request.files['file']
        f.save(secure_filename(f.filename))
        #image = cv2.imread(f.filename)
        results = img_classification(f.filename)
        os.remove(f.filename)
        print(results)
    #return render_template("index.html", FesNames = results)
    html = "<ol>"
    tmp_labels = []
    for each in results:
    	html += "<li>{0}</li>".format(find_vn_festival(each))
    	tmp_labels.append(find_vn_festival(each))
    html += "</ol>"
    global three_festivals
    three_festivals = tmp_labels
    #return "<h4>{0}</h4>".format(results)
    #print(three_festivals)
    return html

#================================
#Take Position:
@app.route("/getAddress", methods=["POST"])
def GetPosition():
    # initialize Nominatim API
    geolocator = Nominatim(user_agent="geoapiExercises")

    lon = request.form["lon"]
    lat = request.form["lat"]
    print("LONLAT:"+lon+" and "+lat)
    Latitude = lat#"50.2989526"
    Longitude = lon#"2.8093828"
      
    location = geolocator.reverse(Latitude+","+Longitude)
      
    address = location.raw['address']
      
    # traverse the data
    city = address.get('city', '')
    state = address.get('state', '')
    country = address.get('country', '')
    code = address.get('country_code')
    zipcode = address.get('postcode')
    print('City : ', city)
    print('State : ', state)
    print('Country : ', country)
    print('Zip Code : ', zipcode)
    
    return "{0}, {1}, {2}, {3}".format(city, state, country, zipcode)
#=====================================
   
if __name__ == '__main__':
   app.run(debug = True)
