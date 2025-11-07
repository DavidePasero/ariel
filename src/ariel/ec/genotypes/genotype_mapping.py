from ariel.ec.genotypes.tree.tree_genome import TreeGenome
from ariel.ec.genotypes.lsystem.l_system_genotype import LSystemDecoder
from ariel.ec.genotypes.cppn.cppn_genome import CPPN_genotype

GENOTYPES_MAPPING = {
    'tree': TreeGenome,
    'lsystem': LSystemDecoder,
    'cppn': CPPN_genotype
}