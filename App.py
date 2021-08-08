from threading import Condition
from flask import Flask , jsonify , request
import flask
from flask_pymongo import PyMongo
import json
import os
from datetime import date, datetime
import shutil
import pymongo
import rasterio
import geopandas as gpd
from geotools.geomask import GeoMask 

# Current date time in local system



# will be invoked from the enpoint where the operator send the confirmation of the file being in the Classified_tiles
# out_tiles = {
#     id : can be from us or generated from the pyMnogo ,
#     path : sting,
#     boundries : array
#     Data : Date
# }

# will be seeded in the database each shape file will be in the shapefile_city original ones (will be filled using script)
# shapeFile_original = {
#     id : can be from us or generated from the pyMnogo ,
#     name : string ,
#     path : string,
#     boundries : array,
#     Date : Date
# }

        
    # 1 x M


# will be invoked after the part of the code that returnes each percentage of each shapeFile
# shapeFile_classified = {
#     id : can be from us or generated from the pyMnogo ,
#     name : string ,
#     path : string,
#     Original_shape_id : id of shapeFile_original,
#     boudnies L array,
#     resource : {dictionary of resoruces , key is the resource name and the value is the m^2}
#     date : Date
# }

# will be seeded in the database each shape file will be in the shapefile_in_city original ones (will be filled using script)
# shapeFile_Child_original = {
#     id : can be from us or generated from the pyMnogo ,
#     name : string ,
#     Original_shape_id : id of shapeFile_original,
#     path : string,
#     boundries : array,
#     Date : Date
# }

# shapeFile_Child_classified = {
#     id : can be from us or generated from the pyMnogo ,
#     name : string ,
#     shapeFile_classified_id : id of shapeFile_classified,
#     path : string,
#     boundries : array,
#     Date : Date
#     resource : {dictionary of resoruces , key is the resource name and the value is the m^2}
# }



def checkBound(shapeFile_bounds , tile_bounds):
    return  shapeFile_bounds[0] > tile_bounds[0] and \
            shapeFile_bounds[1] > tile_bounds[1] and \
            shapeFile_bounds[2] < tile_bounds[2] and \
            shapeFile_bounds[3] < tile_bounds[3] 

app = Flask(__name__)
app.config["MONGO_URI"] = "mongodb://localhost:27017/DataBase"
mongo = PyMongo(app)


@app.route("/savedata" , methods = ["POST"])
def saveData():
    # we can make a Get request to view a form page on request.
    if(request.method == "POST"):
        classified_path = "Data\Classified_tiles"
        fileName = request.form.get("file_name")
        fullpath = os.path.join(os.getcwd() , classified_path)
        FileIsThere = os.path.isfile(os.path.join(fullpath , fileName))
        if(FileIsThere):
            Date = datetime.date(datetime.now())
            folderIsThere = os.path.isdir(os.path.join(fullpath , str(Date)))
            if(not(folderIsThere)):
                os.mkdir(os.path.join(fullpath , str(Date)))
            newPath = shutil.copy(os.path.join(fullpath , fileName), os.path.join(fullpath , str(Date) , fileName))
            os.remove(os.path.join(fullpath , fileName))
            rasterBand = rasterio.open(os.path.join(fullpath , str(Date) , fileName))
            bound_data = list(rasterBand.bounds)
            retunred_object = mongo.db.out_tiles.find_one({"name" : fileName , "date" : str(Date)})
            if(retunred_object == None):
                db_response = mongo.db.out_tiles.insert_one({
                    "path" : str(os.path.join(fullpath , str(Date) , fileName)),
                    "name" : fileName,
                    "bounds" : bound_data,
                    "date" : str(Date)
                })
                print(db_response)

                for i in mongo.db.shapeFile_original.find():
                    print(i)
                    condition = checkBound(i["boundries"] , bound_data)
                    print(i["path"])
                    if(condition):
                        tile_path = os.path.join(fullpath , str(Date) , fileName)
                        obj = GeoMask(tile_path,i["path"])
                        stats , path , name = obj.mask()
                        mongo.db.shapeFile_classified.insert_one({
                            "name" : name.split(".")[0],
                            "path" : path,
                            "Original_shape_id" : i["_id"],
                            "boundries" :i["boundries"],
                            "resource" : stats,
                            "Date" :   str(datetime.date(datetime.now()))
                        })

            if(newPath):
                return json.dumps({'success':True}), 200, {'ContentType':'application/json'} 
            else:
                return 'internal server error', 500
        else:
            return 'bad request!', 400


@app.route("/Resouces/<name>" , methods = ["GET"])
def GetResorces(name):
    data = None
    for i in mongo.db.shapeFile_classified.find({"name" : name}).sort([("Date" , pymongo.DESCENDING)]):
        data = i
        print(data)
        break
    if(data == None):
         return 'not Found !', 404
    else:
        data.pop("_id",None)
        data.pop("Original_shape_id",None)
        print(data)
        return flask.jsonify(data)


@app.route("/Resouces/<name>/<date>" , methods = ["GET"])
def GetResorces_by_date(name,date):
    date_string = str(date)
    print(date_string)
    format = "%Y-%m-%d"
    try:
        datetime.strptime(date_string, format)
    except:
        return 'bad request!', 400
    
    data = mongo.db.shapeFile_classified.find_one({"name" :name , "Date" : str(date)})
    data.pop("_id",None)
    data.pop("Original_shape_id",None)
    print(data)
    
    return flask.jsonify(data)


@app.route("/Resouces/all" , methods = ["GET"])
def Get_all_Resorces():
    data = []
    for i in mongo.db.shapeFile_classified.find().sort([("Date" , pymongo.DESCENDING)]):
        i.pop("_id",None)
        i.pop("Original_shape_id",None)
        data.append(i)
    print(data)
    return flask.jsonify(data)


@app.route("/Date/<name>" , methods = ["GET"])
def Get_City_Dates(name):
    Dates = []
    for i in mongo.db.shapeFile_classified.find({"name" : name}).sort([("Date" , pymongo.ASCENDING)]):
        Dates.append(i["Date"])
    return flask.jsonify(Dates)


@app.route("/Date/all" , methods = ["GET"])
def Get_all_Dates():
    Dates = {}
    for i in mongo.db.shapeFile_classified.find().sort([("Date" , pymongo.ASCENDING)]):
        if(i["name"] in Dates):
            Dates[i["name"]].append(i["Date"])
        else:
            Dates[i["name"]] = []
            Dates[i["name"]].append(i["Date"])
        
    return flask.jsonify(Dates)
