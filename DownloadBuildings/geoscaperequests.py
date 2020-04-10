
from requests import get


def get_building_ids(apikey, coordinates, radius=50):
    """
    Get Building ID's from Geoscape
    
    Parameters
    ----------
    apikey : str
        Geoscape developer ApiKey
    coordinates : str
        lat, lon to query for building
    radius : int, default=50
        distance in meters around coordinates to get buildings
    
    Returns
    -------
    GeoDataFrame
    """
    
    # the geoscape api path
    url = f'https://api.psma.com.au/v1/buildings'
    
    # construct response data
    headers = {'Authorization': apikey}
    params = {'include': 'all',
            'perPage': 100,
            'latLong': coordinates,
            'radius': radius}
    
    # requests building id's
    r = get(url=url, headers=headers, params=params)

    # get all builidng id's
    buildingIds = []
    for building in r.json()['data']:
        buildingIds.append(building['buildingId'])
    
    return buildingIds


def get_building_feature(apikey, buildingId):
    """
    Downloads building data from Geoscape and convert to GeoJSON feature
    
    Parameters
    ----------
    apikey : str
        Geoscape developer ApiKey
    buildingId : str
        Geoscape BuildingId
    """
    # the geoscape api path
    url = f'https://api.psma.com.au/v1/buildings/{buildingId}'

    # construct response data
    headers = {'Authorization': apikey}
    params = {'include': 'all'}
    
    # requests building data
    r = get(url=url, headers=headers, params=params)
    
    # parse information to readable format
    response = r.json()
    
    # convert data to geojson feature
    feature = {}
    feature['type'] = 'Feature'
    feature['geometry'] = response['footprint2d']
    props = ['area', 'averageEaveHeight', 'buildingId',
            'coverageType', 'elevation', 'estimatedLevels',
            'maximumRoofHeight', 'relatedAddressIds',
            'roofComplexity', 'roofMaterial', 'solarPanel',
            'swimmingPool', 'zonings']
    properties = {}
    for prop in props:
        properties[prop] = response[prop]
    properties['centreLon'] = response['centroid']['point']['coordinates'][0]
    properties['centreLat'] = response['centroid']['point']['coordinates'][1]
    feature['properties'] = properties

    return feature