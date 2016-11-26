#!/usr/bin/python

import sys
import csv
import codecs
import pprint
import re
import xml.etree.cElementTree as ET

OSM_PATH = sys.argv[1]
# print( 'The orginal map data:', OSM_PATH )
print 'The orginal map data:', OSM_PATH 

NODES_PATH = "nodes.csv"
NODE_TAGS_PATH = "nodes_tags.csv"
WAYS_PATH = "ways.csv"
WAY_NODES_PATH = "ways_nodes.csv"
WAY_TAGS_PATH = "ways_tags.csv"

LOWER_COLON = re.compile(r'^([a-z]|_)+:([a-z]|_)+')
PROBLEMCHARS = re.compile(r'[=\+/&<>;\'"\?%#$@\,\. \t\r\n]')

NODE_FIELDS = ['id', 'lat', 'lon', 'user', 'uid', 'version', 'changeset', 'timestamp']
NODE_TAGS_FIELDS = ['id', 'key', 'value', 'type']
WAY_FIELDS = ['id', 'user', 'uid', 'version', 'changeset', 'timestamp']
WAY_TAGS_FIELDS = ['id', 'key', 'value', 'type']
WAY_NODES_FIELDS = ['id', 'node_id', 'position']


def shape_element(element, node_attr_fields=NODE_FIELDS, way_attr_fields=WAY_FIELDS,
                  problem_chars=PROBLEMCHARS, default_tag_type='regular'):
    """Clean and shape node or way XML element to Python dict"""

    node_attribs = {}
    way_attribs = {}
    way_nodes = []
    tags = []

    # store node information
    if element.tag == 'node':
        for attr in element.attrib:
            if attr in NODE_FIELDS:
                node_attribs[attr] = element.attrib[attr]
        # store nodes_tags information
        for child in element:
            if child.tag == 'tag' and 'k' in child.attrib and 'v' in child.attrib and not PROBLEMCHARS.match( child.attrib['k'] ):
                tag_dict = {}
                k_value = child.attrib['k'] 
                if LOWER_COLON.match( k_value ):
                    k_value_group = k_value.split(':')
                    tag_dict['type'] = k_value_group[0]
                    k_value_group.pop(0)
                    tag_dict['key'] = ":".join(k_value_group)
                else:
                    tag_dict['type'] = 'regular'
                    tag_dict['key'] = child.attrib['k']
                    
                tag_dict['id'] = element.attrib['id']
                tag_dict['value'] = child.attrib['v']

                # fix subrub names
                if tag_dict['type'] == 'addr' and tag_dict['key'] == 'city':
                    tag_dict['value'] = fix_suburb_names( tag_dict['value'] )
                # fix postcodes
                if tag_dict['type'] == 'addr' and tag_dict['key'] == 'postcode':
                    tag_dict['value'] = fix_postcodes( tag_dict['value'] )
                # fix street name
                if tag_dict['type'] == 'addr' and tag_dict['key'] == 'street':
                    tag_dict['value'] = fix_street_name( tag_dict['value'] )
                if tag_dict['type'] == 'addr' and tag_dict['key'] == 'street' and tag_dict['value'] in ['Shaw', 'Wolli', 'Edward', 'Berith']:
                    tag_dict['value'] = tag_dict['value'] + ' Street'

                tags.append( tag_dict )
                
        return {'node': node_attribs, 'node_tags': tags}
    
    # store way information
    if element.tag == 'way':
        for attr in element.attrib:
            if attr in WAY_FIELDS:
                way_attribs[attr] = element.attrib[attr]
        
        nd_count = 0 # counter of 'nd' tags
        # store ways_tags information
        for child in element:
            if child.tag == 'tag' and 'k' in child.attrib and 'v' in child.attrib and not PROBLEMCHARS.match( child.attrib['k'] ):
                tag_dict = {}
                k_value = child.attrib['k'] 
                if LOWER_COLON.match( k_value ):
                    k_value_group = k_value.split(':')
                    tag_dict['type'] = k_value_group[0]
                    k_value_group.pop(0)
                    tag_dict['key'] = ":".join(k_value_group)
                else:
                    tag_dict['type'] = 'regular'
                    tag_dict['key'] = child.attrib['k']
                    
                tag_dict['id'] = element.attrib['id']
                tag_dict['value'] = child.attrib['v']

                # fix subrub names
                if tag_dict['type'] == 'addr' and tag_dict['key'] == 'city':
                    tag_dict['value'] = fix_suburb_names( tag_dict['value'] )
                # fix postcodes
                if tag_dict['type'] == 'addr' and tag_dict['key'] == 'postcode':
                    tag_dict['value'] = fix_postcodes( tag_dict['value'] )
                # fix street name
                if tag_dict['type'] == 'addr' and tag_dict['key'] == 'street':
                    tag_dict['value'] = fix_street_name( tag_dict['value'] )
                if tag_dict['type'] == 'addr' and tag_dict['key'] == 'street' and tag_dict['value'] in ['Shaw', 'Wolli', 'Edward', 'Berith']:
                    tag_dict['value'] = tag_dict['value'] + ' Street'

                tags.append(tag_dict)

            # store ways_nodes information
            if child.tag == 'nd':
                nd_dict = {}
                nd_dict['id'] = element.attrib['id']
                nd_dict['node_id'] = child.attrib['ref']
                nd_dict['position'] = nd_count
                way_nodes.append(nd_dict)
                nd_count += 1

        return {'way': way_attribs, 'way_nodes': way_nodes, 'way_tags': tags}


# ================================================== #
#               Helper Functions                     #
# ================================================== #
def get_element(osm_file, tags=('node', 'way', 'relation')):
    """Yield element if it is the right type of tag"""

    context = ET.iterparse(osm_file, events=('start', 'end'))
    _, root = next(context)
    for event, elem in context:
        if event == 'end' and elem.tag in tags:
            yield elem
            root.clear()

# fix suburb names
def fix_suburb_names(name_string):
    name_list = name_string.split(' ')
    name_tmp = []
    for name in name_list:
        name_tmp.append( name[0].upper() + name[1:].lower() )
    return ' '.join(name_tmp)

# fix postcodes
def fix_postcodes(postcode):
    nsw_re = re.compile( r'NSW' )
    if nsw_re.search( str(postcode) ): 
        return postcode.split(' ')[1]
    else:
        return postcode

# fix street names
name_mapping = { "St": "Street", "St.": "Street", "st": "Street", "street": "Street", "Ave": "Avenue", "Av": "Avenue", "Rd.": "Road", "Rd": "Road", "road": "Road", "Pl": "Place"}

def fix_street_name(name_string):
    street_type = name_string.split(' ')[-1]
    if street_type in name_mapping.keys():
        return name_string[:-len(street_type)] + name_mapping[street_type]
    else:
        return name_string

class UnicodeDictWriter(csv.DictWriter, object):
    """Extend csv.DictWriter to handle Unicode input"""

    def writerow(self, row):
        super(UnicodeDictWriter, self).writerow( {k: (v.encode('utf-8') if isinstance(v, unicode) else v) for k, v in row.iteritems()} )

    def writerows(self, rows):
        for row in rows:
            self.writerow(row)

# ================================================== #
#               Main Function                        #
# ================================================== #
def process_map(file_in, validate): 
    """Iteratively process each XML element and write to csv(s)"""

    with codecs.open(NODES_PATH, 'w') as nodes_file, \
         codecs.open(NODE_TAGS_PATH, 'w') as nodes_tags_file, \
         codecs.open(WAYS_PATH, 'w') as ways_file, \
         codecs.open(WAY_NODES_PATH, 'w') as way_nodes_file, \
         codecs.open(WAY_TAGS_PATH, 'w') as way_tags_file:

        nodes_writer = UnicodeDictWriter(nodes_file, NODE_FIELDS)
        node_tags_writer = UnicodeDictWriter(nodes_tags_file, NODE_TAGS_FIELDS)
        ways_writer = UnicodeDictWriter(ways_file, WAY_FIELDS)
        way_nodes_writer = UnicodeDictWriter(way_nodes_file, WAY_NODES_FIELDS)
        way_tags_writer = UnicodeDictWriter(way_tags_file, WAY_TAGS_FIELDS)

        nodes_writer.writeheader()
        node_tags_writer.writeheader()
        ways_writer.writeheader()
        way_nodes_writer.writeheader()
        way_tags_writer.writeheader()

        for element in get_element(file_in, tags=('node', 'way')):
            el = shape_element(element)
            if el:
                if element.tag == 'node':
                    nodes_writer.writerow(el['node'])
                    node_tags_writer.writerows(el['node_tags'])
                elif element.tag == 'way':
                    ways_writer.writerow(el['way'])
                    way_nodes_writer.writerows(el['way_nodes'])
                    way_tags_writer.writerows(el['way_tags'])


if __name__ == '__main__':
    process_map(OSM_PATH, validate=True)
