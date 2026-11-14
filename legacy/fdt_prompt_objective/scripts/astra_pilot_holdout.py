"""Deterministic, newly authored evaluation cases; gold is never sent to decoding."""


def make_holdout():
    rows=[]
    names=['Tessa','Ivo','Priya','Leonie','Hugo','Soren']
    cities=['Brackenford','Lindenport','Asterhaven','Northmere','Willowfen','Oakridge']
    for i,name in enumerate(names):
        facts=[
            (f'{name} manages the observatory.',f'{name} is {{value}} years old.',
             f'How old is {name}?',str(37+i*3),str(62+i*2)),
            (f'The shipment for {name} is sealed.',f'Its tracking code is {{value}}.',
             'What is the tracking code?',f'ZX{918437+i*137}',f'BT{603821+i*193}'),
            (f'{name} organized a workshop.',f'The workshop is in {{value}}.',
             'Where is the workshop?',cities[i],cities[(i+3)%6]),
            (f'A storage rack is assigned to {name}.',f'The rack contains {{value}} lenses.',
             'How many lenses are on the rack?',str(23+i*7),str(81+i*4)),
            (f'{name} reserved the telescope.',f'The reservation starts at {{value}}.',
             'When does the reservation start?',f'{10+i:02d}:17',f'{17+i:02d}:43'),
            (f'A service node belongs to {name}.',f'Its listening port is {{value}}.',
             'Which port does the service use?',str(7341+i*23),str(9127+i*29)),
            (f'{name} packed a tool case.',f'The case is {{value}}.',
             'What color is the case?',['violet','orange','silver','brown','teal','white'][i],
             ['green','black','yellow','blue','red','pink'][i]),
            (f'{name} signed an invoice.',f'The invoice total is {{value}} dollars.',
             'What is the invoice total in dollars?',str(247+i*38),str(813+i*26)),
        ]
        for j,(context,template,question,old,new) in enumerate(facts):
            for variant,value in [('original',old),('changed',new),('omitted',None)]:
                span=template.format(value=value) if value is not None else ''
                prompt=f'Record: {context} {span}\nQuestion: {question}\nAnswer:'
                rows.append({'id':f'fact_{i:02d}_{j}_{variant}','family':'fact',
                    'pair_id':f'fact_{i:02d}_{j}','variant':variant,'prompt':prompt,
                    'expected':value,'alternatives':[old,new],'evidence_span':span or None})
    natural=[
        'The potter left a bowl to dry overnight. In the morning, a crack along the rim suggested',
        'A ferry cannot carry every waiting car at once. The crew therefore needs to',
        'The museum label gave a date but no information about the artist. A visitor who wanted to learn more could',
        'A small town replaced its broken streetlights. Residents walking home after sunset noticed',
        'Before lending her compass to a friend, the hiker explained how to',
        'The baker measured the same ingredients on two scales. When the results differed, she first checked',
        'A volunteer found a damaged book in the return box. To prevent further damage,',
        'The sound of the waterfall grew louder as the path descended. Around the next bend,',
        'The curtains moved even though the door was closed. A possible explanation was',
        'A cracked pipe was leaking into the cellar. The homeowner first needed to',
        'The class planted beans in two pots, watering only one. After a week they compared',
        'An umbrella keeps rain off a person by',
        'A thermometer left in sunlight can give a misleading air temperature because',
        'A shadow becomes longer when the light source moves lower because',
        'The same journey takes longer at a lower average speed because',
        'A sealed jar can preserve food by reducing contact with',
        'The carpenter measured the doorway before cutting the panel so that',
        'The explorer marked the fork in the trail to help the group',
        'When a bus route changes, passengers need clear information about',
        'Two witnesses described the same event differently. An investigator should compare',
        'The editor shortened the paragraph while keeping its main point. This required',
        'A fair test changes one factor while keeping the others stable so that',
        'The shop counted its inventory twice after finding a discrepancy. The second count would help',
        'A community garden needs a watering schedule because',
        'The pianist slowed down the difficult passage during practice in order to',
        'A locked archive protects original records, but researchers still need a way to',
        'The translator encountered a word with several meanings. The surrounding sentence helped',
        'A campsite should be left clean so that the next visitors can',
        'The repair manual listed symptoms before possible causes. This organization helps readers',
        'A boat carrying too much weight sits lower in the water because',
        'The team recorded an unexpected result instead of discarding it. Repeating the experiment could',
        'A calendar reminder is useful only if it includes enough information to',
    ]
    rows.extend({'id':f'natural_{i:02d}','family':'natural','prompt':p,'expected':None}
                for i,p in enumerate(natural))
    structured=[
        ('Record: owner=Tessa, count=23. Convert the record to JSON.\nJSON:', '{"owner":"Tessa","count":23}'),
        ('Record: active=false, port=7341. Convert the record to JSON.\nJSON:', '{"active":false,"port":7341}'),
        ('Record: city=Lindenport, color=orange. Convert the record to JSON.\nJSON:', '{"city":"Lindenport","color":"orange"}'),
        ('Record: price=247, currency=USD. Convert the record to JSON.\nJSON:', '{"price":247,"currency":"USD"}'),
        ('Write a JSON array containing the numbers 7, 11, and 19.\nJSON:', '[7,11,19]'),
        ('Record: enabled=true, retries=4. Convert the record to JSON.\nJSON:', '{"enabled":true,"retries":4}'),
        ('Write a JSON object with key "note" and a null value.\nJSON:', '{"note":null}'),
        ('Record: code=ZX918437, paid=true. Convert the record to JSON.\nJSON:', '{"code":"ZX918437","paid":true}'),
        ('Complete this Python function to return the square of n.\ndef square(n):\n', '    return n * n'),
        ('Complete this Python function to test whether n is even.\ndef is_even(n):\n', '    return n % 2 == 0'),
        ('Complete this Python function to return the first item of a nonempty list.\ndef first(items):\n', '    return items[0]'),
        ('Complete this Python function to count items in a list.\ndef count_items(items):\n', '    return len(items)'),
        ('Complete this Python function to join two strings.\ndef join_strings(left, right):\n', '    return left + right'),
        ('Complete this Python function to return an empty dictionary.\ndef empty_mapping():\n', '    return {}'),
        ('Complete this Python function to subtract b from a.\ndef difference(a, b):\n', '    return a - b'),
        ('Complete this Python function to check whether a list is empty.\ndef is_empty(items):\n', '    return len(items) == 0'),
    ]
    rows.extend({'id':f'structured_{i:02d}','family':'structured','prompt':p,'expected':g}
                for i,(p,g) in enumerate(structured))
    assert len(rows)==192 and len({r['id'] for r in rows})==192
    return rows
