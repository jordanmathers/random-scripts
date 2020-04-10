import PySimpleGUI as sg
import json

from geoscaperequests import get_building_ids, get_building_feature

sg.theme('SystemDefault')

layout = [[sg.Text('ApiKey'), sg.InputText()],
         [sg.Text('Coordinates'), sg.InputText()],
         [sg.Text('Radius(m) 1-100'), sg.InputText('50')],
         [sg.Input(), sg.FileSaveAs()],
         [sg.OK(), sg.Cancel()]]

window = sg.Window('Download Geoscape Buildings', layout)

event, values = window.Read()

window.Close()

apikey = values[0]
coordiantes = values[1]
radius = values[2]
filepath = values[3]

if event == 'OK':
    buildingIds = get_building_ids(values[0], values[1], values[2])

layout = [[sg.Text(f'This location has {len(buildingIds)} buildings')],
          [sg.Text('Would you like to continue?')],
          [sg.OK(), sg.Cancel()]]

window = sg.Window('Download Geoscape Buildings', layout)

event, values  = window.Read()

window.Close()

if event == 'OK':
    features = []
    for i in range(len(buildingIds)):
        features.append(get_building_feature(apikey, buildingIds[i]))
        sg.OneLineProgressMeter('Downloading Buildings', i+1, len(buildingIds), 'key', orientation='h')
    feature_collection = {'type': 'FeatureCollection',
                          'features': features}
    with open(f'{filepath}.geojson', 'w') as f:
        json.dump(feature_collection, f)
    sg.Popup('Download Complete!')