def get_plant_info(plant_name):
    """Returns information about the identified plant"""
    plant_database = {
        'Tulsi': {
            'common_name': 'Holy Basil',
            'scientific_name': 'Ocimum sanctum',
            'uses': [
                'Treats respiratory disorders',
                'Reduces stress and anxiety',
                'Boosts immunity',
                'Anti-inflammatory properties'
            ]
        },
        'Neem': {
            'common_name': 'Indian Lilac',
            'scientific_name': 'Azadirachta indica',
            'uses': [
                'Natural antiseptic',
                'Treats skin disorders',
                'Dental care',
                'Blood purifier'
            ]
        },
        'Aloe Vera': {
            'common_name': 'Aloe Vera',
            'scientific_name': 'Aloe barbadensis miller',
            'uses': [
                'Skin healing',
                'Burns treatment',
                'Digestive aid',
                'Anti-inflammatory'
            ]
        },
        'Mint': {
            'common_name': 'Peppermint',
            'scientific_name': 'Mentha × piperita',
            'uses': [
                'Digestive aid',
                'Relieves headaches',
                'Fresh breath',
                'Reduces nausea'
            ]
        }
    }
    
    return plant_database.get(plant_name, {
        'common_name': 'Unknown',
        'scientific_name': 'Unknown',
        'uses': ['Information not available']
    })
