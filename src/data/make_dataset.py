# -*- coding: utf-8 -*-
import click
import logging
import json
import pandas as pd
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from lxml import etree


def normalize_pos(category):
    """Normalize POS tags to standard format."""
    pos_mapping = {
        'adj': 'ADJ',
        'nomen': 'NOUN', 
        'subst': 'NOUN',
        'verb': 'VERB',
        'verben': 'VERB'
    }
    return pos_mapping.get(category.lower(), category.upper())


def parse_germanet_xml_files(xml_dir):
    """Parse all GermaNet XML files to extract synsets and lexical units."""
    xml_dir = Path(xml_dir)
    synsets = []
    lexical_units = []
    
    logger = logging.getLogger(__name__)
    logger.info(f'Parsing XML files from {xml_dir}')
    
    # Skip DTD, InterLingualIndex, and relations files
    skip_patterns = ['gn_relations', 'interlingualindex', '.dtd', 'germanet_']
    xml_files = [f for f in xml_dir.glob('*.xml') 
                 if not any(pattern in f.name.lower() for pattern in skip_patterns)]
    
    logger.info(f'Found {len(xml_files)} XML files to process')
    
    for xml_file in xml_files:
        logger.info(f'Processing {xml_file.name}')
        
        try:
            # Use forgiving parser for malformed XML
            parser = etree.XMLParser(recover=True, huge_tree=True)
            tree = etree.parse(str(xml_file), parser)
            root = tree.getroot()
            
            file_synsets = 0
            file_lexunits = 0
            
            for synset in root.xpath('//synset'):
                synset_id = synset.get('id')
                category = synset.get('category')
                synset_class = synset.get('class')
                normalized_pos = normalize_pos(category) if category else None
                
                # Extract paraphrase if present
                paraphrase_elem = synset.find('paraphrase')
                paraphrase = paraphrase_elem.text if paraphrase_elem is not None else None
                
                synset_data = {
                    'synset_id': synset_id,
                    'pos': normalized_pos,
                    'class': synset_class,
                    'paraphrase': paraphrase
                }
                synsets.append(synset_data)
                file_synsets += 1
                
                # Extract lexical units for this synset
                for lexunit in synset.xpath('lexUnit'):
                    lexunit_id = lexunit.get('id')
                    sense = lexunit.get('sense')
                    source = lexunit.get('source')
                    named_entity = lexunit.get('namedEntity') == 'yes'
                    artificial = lexunit.get('artificial') == 'yes'
                    style_marking = lexunit.get('styleMarking') == 'yes'
                    
                    # Extract orthographic form (lemma)
                    orth_form = lexunit.find('orthForm')
                    lemma = orth_form.text if orth_form is not None else None
                    
                    # Extract ALL variants as list
                    variants = []
                    for orth_var in lexunit.xpath('orthVar'):
                        if orth_var.text:
                            variants.append(orth_var.text)
                    
                    # Extract ALL examples as list
                    examples = []
                    for example in lexunit.xpath('example/text'):
                        if example.text:
                            examples.append(example.text)
                    
                    lexunit_data = {
                        'lexunit_id': lexunit_id,
                        'synset_id': synset_id,
                        'lemma': lemma,
                        'variants': variants,
                        'named_entity': named_entity,
                        'artificial': artificial,
                        'style_marking': style_marking,
                        'examples': examples,
                        'pos': normalized_pos,
                        'class': synset_class,
                        'sense': sense,
                        'source': source
                    }
                    lexical_units.append(lexunit_data)
                    file_lexunits += 1
                    
            logger.info(f'{xml_file.name}: {file_synsets} synsets, {file_lexunits} lexical units')
                    
        except Exception as e:
            logger.error(f'Error processing {xml_file.name}: {e}')
            continue
    
    return synsets, lexical_units


def parse_relations(relations_file):
    """Parse the semantic relations from gn_relations.xml."""
    logger = logging.getLogger(__name__)
    logger.info(f'Parsing relations from {relations_file}')
    
    relations = []
    
    try:
        parser = etree.XMLParser(recover=True, huge_tree=True)
        tree = etree.parse(str(relations_file), parser)
        root = tree.getroot()
        
        for rel in root.xpath('//con_rel'):
            relation_data = {
                'relation_type': rel.get('name'),
                'from_synset': rel.get('from'),
                'to_synset': rel.get('to'),
                'direction': rel.get('dir'),
                'inverse': rel.get('inv')
            }
            relations.append(relation_data)
            
    except Exception as e:
        logger.error(f'Error parsing relations: {e}')
    
    return relations


def create_lemma_senses_table(synsets_df, lexical_units_df):
    """Create lemma_senses table by joining lexical units with synsets."""
    logger = logging.getLogger(__name__)
    logger.info('Creating lemma_senses table')
    
    # Join lexical units with synsets on synset_id
    lemma_senses = lexical_units_df.merge(
        synsets_df[['synset_id', 'paraphrase']], 
        on='synset_id', 
        how='left'
    )
    
    # Select and rename columns for final output
    lemma_senses = lemma_senses[[
        'lemma', 'pos', 'synset_id', 'class', 'paraphrase', 
        'variants', 'examples'
    ]].copy()
    
    # Remove duplicates (same lemma-synset combinations)
    lemma_senses = lemma_senses.drop_duplicates(subset=['lemma', 'synset_id'])
    
    logger.info(f'Created {len(lemma_senses)} unique lemma-synset pairs')
    return lemma_senses


def save_dataframes(output_dir, synsets_df, lexical_units_df, relations_df, lemma_senses_df, 
                   save_csv=False, save_json=False):
    """Save dataframes as Parquet (primary) and optionally CSV/JSON."""
    logger = logging.getLogger(__name__)
    
    # Create germanet_v190 subdirectory
    germanet_dir = output_dir / 'germanet_v190'
    germanet_dir.mkdir(parents=True, exist_ok=True)
    
    tables = {
        'synsets': synsets_df,
        'lexical_units': lexical_units_df, 
        'relations': relations_df,
        'lemma_senses': lemma_senses_df
    }
    
    for name, df in tables.items():
        logger.info(f'Saving {name}: {len(df)} rows')
        
        # Primary format: Parquet
        try:
            df.to_parquet(germanet_dir / f'{name}.parquet', index=False)
        except Exception as e:
            logger.error(f'Failed to save {name}.parquet: {e}')
        
        # Optional formats
        if save_csv:
            try:
                # For CSV, join lists with | separator  
                df_csv = df.copy()
                for col in df_csv.columns:
                    if df_csv[col].dtype == 'object':
                        df_csv[col] = df_csv[col].apply(
                            lambda x: '|'.join(x) if isinstance(x, list) and x else (x if x is not None else '')
                        )
                df_csv.to_csv(germanet_dir / f'{name}.csv', index=False, encoding='utf-8')
            except Exception as e:
                logger.error(f'Failed to save {name}.csv: {e}')
        
        if save_json:
            try:
                df.to_json(germanet_dir / f'{name}.json', orient='records', 
                          indent=2, ensure_ascii=False)
            except Exception as e:
                logger.error(f'Failed to save {name}.json: {e}')
    
    logger.info(f'Data saved to {germanet_dir}')


@click.command()
@click.argument('input_root', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path())
@click.option('--csv', is_flag=True, help='Also save CSV files')
@click.option('--json', is_flag=True, help='Also save JSON files')
def main(input_root, output_dir, csv, json):
    """Process GermaNet data from XML files to structured format.
    
    INPUT_ROOT: Root directory containing GN_V190/GN_V190_XML/
    OUTPUT_DIR: Directory to save processed data
    """
    logger = logging.getLogger(__name__)
    logger.info('Processing GermaNet data from raw XML files')
    
    input_path = Path(input_root)
    output_path = Path(output_dir)
    
    # Resolve XML directory
    germanet_xml_dir = input_path / 'GN_V190' / 'GN_V190_XML'
    
    if not germanet_xml_dir.exists():
        logger.error(f'GermaNet XML directory not found: {germanet_xml_dir}')
        return
    
    # Parse synsets and lexical units
    synsets, lexical_units = parse_germanet_xml_files(germanet_xml_dir)
    logger.info(f'Parsed {len(synsets)} synsets and {len(lexical_units)} lexical units')
    
    # Parse relations
    relations_file = germanet_xml_dir / 'gn_relations.xml'
    relations = parse_relations(relations_file)
    logger.info(f'Parsed {len(relations)} semantic relations')
    
    # Convert to DataFrames
    synsets_df = pd.DataFrame(synsets)
    lexical_units_df = pd.DataFrame(lexical_units)
    relations_df = pd.DataFrame(relations)
    
    # Create lemma_senses table
    lemma_senses_df = create_lemma_senses_table(synsets_df, lexical_units_df)
    
    # Save all tables
    save_dataframes(output_path, synsets_df, lexical_units_df, relations_df, 
                   lemma_senses_df, save_csv=csv, save_json=json)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()