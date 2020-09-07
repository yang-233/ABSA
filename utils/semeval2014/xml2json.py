import json
import re
import sys
import xml.etree.ElementTree as ET

lap_train_filename = "./data/semeval2014/Laptop_Train_v2.xml"
rest_train_filename = "./data/semeval2014/Restaurants_Train_v2.xml"

lap_test_filename = "./data/semeval2014/Laptops_Test_Gold.xml"
rest_test_filename = "./data/semeval2014/Restaurants_Test_Gold.xml"

build_aspect = lambda e: {"term" : e.attrib["term"], "polarity" : e.attrib["polarity"], 
                                                        "from" : e.attrib["from"], "to" : e.attrib["to"]}
build_category = lambda e: {"category" : e.attrib["category"], "polarity" : e.attrib["polarity"]}
build_sentence = lambda s: {"id" : s.attrib["id"],
                            "text" : s.find("text").text, 
                            "aspectTerms" : [build_aspect(e) for es in s.findall('aspectTerms') 
                                            for e in es if es is not None],
                            "aspectCategories" : [build_category(e) for es in s.findall('aspectCategories') 
                                                 for e in es if es is not None]
                            }

def semeval2014ToList(filename):
    root = ET.parse(filename).getroot()
    return [build_sentence(s) for s in root.findall("sentence")]
    

def semeval2014ToJson(filename, path="./data/semeval2014/"):
    jsonFileName = path + re.split("[./]", filename)[-2] + ".json"
    with open(jsonFileName, "w") as o:
        json.dump(semeval2014ToList(filename), o, indent=4)

def test():
    root = ET.parse(lap_train_filename).getroot()
    sentences = root.findall("sentence")
    s = sentences[0]
    print(type(s.attrib["id"]))
def main():
    l = semeval2014ToList(lap_train_filename)    
    count = 0
    for i in l:
        #if len(i["aspectTerms"]) == 0:
        #    print(i["id"])
        for j in i["aspectTerms"]:
            if j["polarity"] == "conflict":
                continue
            count += 1
    #print("Done! %d" % count)
    #print(len(l) - count)
    print(count)
if __name__ == "__main__":
    main()