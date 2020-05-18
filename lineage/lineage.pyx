# cython: boundscheck=False
# cython: cdivision=True
# cython: wraparound=False

import numpy as np
from libc.math cimport round
cimport numpy as np
from bioscrape.simulator cimport VolumeCellState, VolumeSSAResult, DelayVolumeSSAResult, VolumeSplitter, CSimInterface, ModelCSimInterface, DelayVolumeSSASimulator, VolumeSSASimulator, DelayQueue, DelayVolumeCellState
from bioscrape.simulator import VolumeCellState, VolumeSSAResult, DelayVolumeSSAResult, VolumeSplitter, CSimInterface, ModelCSimInterface, DelayVolumeSSASimulator, VolumeSSASimulator, DelayQueue, DelayVolumeCellState

from bioscrape.types cimport Model, Volume, Schnitz, Lineage, Propensity, Term, Rule
#from bioscrape.types import Model, Volume, Schnitz, Lineage, Propensity, Term, Rule

from bioscrape.types import sympy_species_and_parameters, parse_expression
from bioscrape.random cimport normal_rv

cimport bioscrape.random as cyrandom
import bioscrape.random as cyrandom

from bioscrape.vector cimport vector 
#from vector cimport vector 

import warnings


#Events are general objects that happen with some internal propensity but are not chemical reactions
cdef class Event:

	cdef initialize(self, dict event_params, dict species_indices, dict parameter_indices):
		raise NotImplementedError("VolumeReactions Must be subclassed")

	def get_species_and_parameters(self, dict event_fields):
		raise NotImplementedError("VolumeReactions Must be subclassed")

	#cdef Propensity get_propensity(self):
	#	return <Propensity>self.propensity

	#Meant to be subclassed if an event is supposed to do something.
	cdef double evaluate_event(self, double* state, double *params, double volume, double time):
		return 0

#Volume Events are stochastic events which alter a single cell's volume.
cdef class VolumeEvent(Event):
	cdef double evaluate_event(self, double* state, double *params, double volume, double time):
		return self.get_volume(state, params,volume, time)

	cdef double get_volume(self, double* state, double *params, double volume, double time):
		raise NotImplementedError("VolumeEvent must be subclassed")

	def get_species_and_parameters(self, dict event_fields):
		raise NotImplementedError("VolumeReactions Must be subclassed")

cdef class LinearVolumeEvent(VolumeEvent):
	cdef unsigned growth_rate_ind

	cdef initialize(self, dict event_params, dict species_indices, dict parameter_indices):
		for (k, v) in event_params.items():
			if k == "growth_rate":
				self.growth_rate_ind = parameter_indices[v]
			else:
				warnings.warn("Useless paramter for LinearVolumeEvent: "+str(k))

	cdef double get_volume(self, double* state, double *params, double volume, double time):
		return volume + params[self.growth_rate_ind]

	def get_species_and_parameters(self, dict event_fields):
		return ([], [event_fields["growth_rate"]])

cdef class MultiplicativeVolumeEvent(VolumeEvent):
	cdef unsigned growth_rate_ind
	cdef Term volume_equation


	cdef initialize(self, dict event_params, dict species_indices, dict parameter_indices):
		for (k, v) in event_params.items():
			if k == "growth_rate":
				self.growth_rate_ind = parameter_indices[v]
			else:
				warnings.warn("Useless paramter for MultiplicativeVolumeEvent: "+str(k))

	cdef double get_volume(self, double* state, double *params, double volume, double time):
		return volume*(1+params[self.growth_rate_ind])

	def get_species_and_parameters(self, dict event_fields):
		return ([], [event_fields["growth_rate"]])

cdef class GeneralVolumeEvent(VolumeEvent):
	cdef Term volume_equation

	cdef initialize(self, dict event_params, dict species_indices, dict parameter_indices):
		for (k, v) in event_params.items():
			if k == "equation":
				self.volume_equation = parse_expression(v, species_indices, parameter_indices)
			else:
				warnings.warn("Useless paramter for GeneralVolumeEvent: "+str(k))

	cdef double get_volume(self, double* state, double *params, double volume, double time):
		return (<Term>self.volume_equation).volume_evaluate(state,params,volume,time)


	def get_species_and_parameters(self, dict event_fields):
		equation_string = event_fields['equation'].strip()
		species_r, parameters_r = sympy_species_and_parameters(equation_string)
		return (species_r, parameters_r)

#Division Events are stochastic events which cause a cell to divide using a particular VolumeSplitter
cdef class DivisionEvent(Event):
	cdef initialize(self, dict event_params, dict species_indices, dict parameter_indices):
		#self.propensity = propensity
		#self.propensity.initialize(propensity_params, species_indices, parameter_indices)
		#self.vsplit = vsplit
		pass

	#cdef Propensity get_propensity(self):
	#	return <Propensity>self.propensity

	#Return 1 indicated the cell divided. 0 indicates it didn't. Could be used in a subclass.
	cdef double evaluate_event(self, double* state, double *params, double volume, double time):
		return 1

	def get_species_and_parameters(self, dict event_fields):
		return ([], [])


#Death Events are stochastic events which casue a cell to die
cdef class DeathEvent(Event):
	cdef initialize(self, dict event_params, dict species_indices, dict parameter_indices):
		pass

	def get_species_and_parameters(self, dict event_fields):
		return ([], [])

	#Could be subclassed to have more complex logic for if a cell dies when this event occurs
	#0 indicates not dead. 1 indicates dead.
	cdef double evaluate_event(self, double* state, double *params, double volume, double time):
		return 1

#Dummy class to help with inheritance, compilation, and code simplification. Does nothing
cdef class LineageRule:
	def initialize(self, dict param_dictionary, dict species_indices, dict parameter_indices):
		raise NotImplementedError("LineageRule must be subclassed")

	def get_species_and_parameters(self, dict fields):
		raise NotImplementedError("LineageRule must be subclassed")

#Volume Rules occur every dt (determined by the simulation timepoints) and update the volume
cdef class VolumeRule(LineageRule):
	cdef double get_volume(self, double* state, double *params, double volume, double time, double dt):
		raise NotImplementedError("get_volume must be implemented in VolumeRule Subclasses")

cdef class LinearVolumeRule(VolumeRule):
	cdef unsigned has_noise
	cdef unsigned growth_rate_ind
	cdef unsigned noise_ind
	cdef double get_volume(self, double* state, double *params, double volume, double time, double dt):
		if self.has_noise > 0:
			return volume + (params[self.growth_rate_ind]+normal_rv(0, params[self.noise_ind]))*dt
		else:
			return volume + params[self.growth_rate_ind]*dt

	def initialize(self, dict param_dictionary, dict species_indices, dict parameter_indices):
		self.has_noise = 0
		for (k, v) in param_dictionary.items():
			if k == "growth_rate":
				self.growth_rate_ind = parameter_indices[v]
			elif k == "noise":
				self.has_noise = 1
				self.noise_ind = parameter_indices[v]
			else:
				warnings.warn("Useless paramter for LinearVolumeRule: "+str(k))
		
	def get_species_and_parameters(self, dict fields):
		if self.has_noise > 0:
			return ([], [fields["growth_rate"], fields["noise"]])
		else:
			return ([], [fields["growth_rate"]])

cdef class MultiplicativeVolumeRule(VolumeRule):
	cdef unsigned has_noise
	cdef unsigned growth_rate_ind
	cdef unsigned noise_ind
	cdef double get_volume(self, double* state, double *params, double volume, double time, double dt):
		if self.has_noise > 0:
			return volume+volume*(params[self.growth_rate_ind]+normal_rv(0, params[self.noise_ind]))*dt
		else:
			return volume+volume*params[self.growth_rate_ind]*dt

	def initialize(self, dict param_dictionary, dict species_indices, dict parameter_indices):
		self.has_noise = 0
		for (k, v) in param_dictionary.items():
			if k == "growth_rate":
				self.growth_rate_ind = parameter_indices[v]
			elif k == "noise":
				self.has_noise = 1
				self.noise_ind = parameter_indices[v]
			else:
				warnings.warn("Useless paramter for MultiplicativeVolumeRule: "+str(k))

	def get_species_and_parameters(self, dict fields):
		if self.has_noise:
			return ([], [fields["growth_rate"], fields['noise']])
		else:
			return ([], [fields["growth_rate"]])

cdef class AssignmentVolumeRule(VolumeRule):
	cdef Term volume_equation
	cdef double get_volume(self, double* state, double *params, double volume, double time, double dt):
		return (<Term>self.volume_equation).volume_evaluate(state,params,volume,time)

	def initialize(self, dict param_dictionary, dict species_indices, dict parameter_indices):
		for (k, v) in param_dictionary.items():
			if k == "equation":
				self.volume_equation = parse_expression(v, species_indices, parameter_indices)
			else:
				warnings.warn("Useless paramter for AssignmentVolumeRule: "+str(k))

	def get_species_and_parameters(self, dict fields):
		equation_string = fields['equation'].strip()
		species, parameters = sympy_species_and_parameters(equation_string)
		return (species, parameters)

cdef class ODEVolumeRule(VolumeRule):
	cdef Term volume_equation
	cdef double get_volume(self, double* state, double *params, double volume, double time, double dt):
		return volume+(<Term>self.volume_equation).volume_evaluate(state,params,volume,time)*dt

	def initialize(self, dict param_dictionary, dict species_indices, dict parameter_indices):
		for (k, v) in param_dictionary.items():
			if k == "equation":
				self.volume_equation = parse_expression(v, species_indices, parameter_indices)
			else:
				warnings.warn("Useless paramter for ODEVolumeRule: "+str(k))

	def get_species_and_parameters(self, dict fields):
		equation_string = fields['equation'].strip()
		species, parameters = sympy_species_and_parameters(equation_string)
		return (species, parameters)

#Division rules are checked at the beginning of each simulation loop to see if a cell should divide
cdef class DivisionRule(LineageRule):

	cdef int check_divide(self, double* state, double *params, double time, double volume, double initial_time, double initial_volume):
		raise NotImplementedError("check_divide must be implemented in DivisionRule subclasses")

	def initialize(self, dict param_dictionary, dict species_indices, dict parameter_indices):
		raise NotImplementedError("DivisionRule must be subclassed!")

	def get_species_and_parameters(self, dict fields):
		raise NotImplementedError("DivisionRule must be subclassed!")

#A division rule where division occurs after some amount of time (with an optional noise term)
cdef class TimeDivisionRule(DivisionRule):
	cdef unsigned has_noise
	cdef unsigned threshold_ind
	cdef unsigned threshold_noise_ind
	cdef int check_divide(self, double* state, double *params, double time, double volume, double initial_time, double initial_volume):
		if self.has_noise > 0:
			if time-initial_time >= params[self.threshold_ind]+normal_rv(0, params[self.threshold_noise_ind]):
				return 1
			else:
				return 0
		else:
			if time-initial_time >= params[self.threshold_ind] - 1E-9:
				return 1
			else:
				return 0

	def initialize(self, dict param_dictionary, dict species_indices, dict parameter_indices):
		for (k, v) in param_dictionary.items():
			if k == "threshold":
				self.threshold_ind = parameter_indices[v]
			elif k == "noise":
				self.threshold_noise_ind = parameter_indices[v]
				self.has_noise = 1
			else:
				warnings.warn("Useless paramter for TimeDivisionRule: "+str(k))
		if "noise" not in param_dictionary.keys():
			self.has_noise = 0

	def get_species_and_parameters(self, dict fields):
		if "noise" in fields:
			return ([], [fields["threshold"], fields["noise"]])
		else:
			return ([], [fields["threshold"]])

#A division rule where division occurs at some volume threshold (with an optional noise term)
cdef class VolumeDivisionRule(DivisionRule):
	cdef unsigned has_noise
	cdef unsigned threshold_ind
	cdef unsigned threshold_noise_ind
	cdef int check_divide(self, double* state, double *params, double time, double volume, double initial_time, double initial_volume):
		if self.has_noise > 0:
			if volume >= params[self.threshold_ind]+normal_rv(0, params[self.threshold_noise_ind]):
				return 1
			else:
				return 0
		else:
			if volume >= params[self.threshold_ind] - 1E-9:
				return 1
			else:
				return 0

	def initialize(self, dict param_dictionary, dict species_indices, dict parameter_indices):
		for (k, v) in param_dictionary.items():
			if k == "threshold":
				self.threshold_ind = parameter_indices[v]
			elif k == "noise":
				self.threshold_noise_ind = parameter_indices[v]
				self.has_noise = 1
			else:
				warnings.warn("Useless paramter for VolumeDivisionRule: "+str(k))
		if "noise" not in param_dictionary.keys():
			self.has_noise = 0

	def get_species_and_parameters(self, dict fields):
		if "noise" in fields:
			return ([], [fields["threshold"], fields["noise"]])
		else:
			return ([], [fields["threshold"]])

#A division rule where division occurs after the cell has grown by some amount delta (with an optional noise term)
cdef class DeltaVDivisionRule(DivisionRule):
	cdef unsigned has_noise
	cdef unsigned threshold_ind
	cdef unsigned threshold_noise_ind
	cdef int check_divide(self, double* state, double *params, double time, double volume, double initial_time, double initial_volume):
		if self.has_noise > 0:
			if volume - initial_volume >= params[self.threshold_ind]+normal_rv(0, params[self.threshold_noise_ind]):
				return 1
			else:
				return 0
		else:
			if volume - initial_volume >= params[self.threshold_ind] - 1E-9:
				return 1
			else:
				return 0

	def initialize(self, dict param_dictionary, dict species_indices, dict parameter_indices):
		for (k, v) in param_dictionary.items():
			if k == "threshold":
				self.threshold_ind = parameter_indices[v]
			elif k == "noise":
				self.threshold_noise_ind = parameter_indices[v]
				self.has_noise = 1
			else:
				warnings.warn("Useless paramter for DeltaVDivisionRule: "+str(k))
		if "noise" not in param_dictionary.keys():
			self.has_noise = 0

	def get_species_and_parameters(self, dict fields):
		if "noise" in fields:
			return ([], [fields["threshold"], fields["noise"]])
		else:
			return ([], [fields["threshold"]])

#A general division rule
#returns 1 if equation(state, params, volume, time) > 0
cdef class GeneralDivisionRule(DivisionRule):
	cdef Term equation
	cdef int check_divide(self, double* state, double *params, double time, double volume, double initial_time, double intial_volume):
		if (<Term> self.equation).volume_evaluate(state,params,volume,time) > 0:
			return 1
		else:
			return 0

	def initialize(self, dict param_dictionary, dict species_indices, dict parameter_indices):
		for (k, v) in param_dictionary.items():
			if k == "equation":
				self.equation = parse_expression(v, species_indices, parameter_indices)
			else:
				warnings.warn("Useless paramter for GeneralDivisionRule: "+str(k))

	def get_species_and_parameters(self, dict fields):
		equation_string = fields['equation'].strip()
		species, parameters = sympy_species_and_parameters(equation_string)
		return (species, parameters)

#Death rules are checked at the beginning of each simulation loop to see if a cell should die
cdef class DeathRule(LineageRule):
	cdef int check_dead(self, double* state, double *params, double time, double volume, double initial_time, double initial_volume):
		raise NotImplementedError("check_dead must be implemented in DeathRule subclasses.")

	def initialize(self, dict param_dictionary, dict species_indices, dict parameter_indices):
		raise NotImplementedError("DeathRule must be Subclassed!")

	def get_species_and_parameters(self, dict fields):
		raise NotImplementedError("DeathRule must be Subclassed!")

#A death rule where death occurs when some species is greater than or less than a given threshold
cdef class SpeciesDeathRule(DeathRule):
	cdef unsigned has_noise
	cdef unsigned species_ind
	cdef unsigned threshold_ind
	cdef unsigned threshold_noise_ind
	cdef int comp
	cdef double threshold
	cdef int check_dead(self, double* state, double *params, double time, double volume, double initial_time, double initial_volume):
		self.threshold = params[self.threshold_ind]
		if self.has_noise > 0:
			self.threshold = self.threshold + normal_rv(0, params[self.threshold_noise_ind])

		if state[self.species_ind] > self.threshold - 1E-9 and state[self.species_ind] < self.threshold + 1E-9 and self.comp == 0:
			return 1
		elif state[self.species_ind] > self.threshold-1E-9 and self.comp == 1:
			return 1
		elif state[self.species_ind] < self.threshold+ 1E-9 and self.comp == -1:
			return 1
		else:
			return 0

	def initialize(self, dict param_dictionary, dict species_indices, dict parameter_indices):
		for (k, v) in param_dictionary.items():
			if k == "specie":
				self.species_ind = species_indices[v]
			elif k == "threshold":
				self.threshold_ind = parameter_indices[v]
			elif k == "noise":
				self.threshold_noise_ind = parameter_indices[v]
				self.has_noise = 1
			elif k == "comp":
				if v == "=" or v =="equal":
					self.comp = 0
				elif v == "<" or v == "less":
					self.comp = -1
				elif v == ">" or v == "greater":
					self.comp = 1
			else:
				warnings.warn("Useless paramter for SpeciesDeathRule: "+str(k))
		if "noise" not in param_dictionary.keys():
			self.has_noise = 0
		if "comp" not in param_dictionary.keys():
			warnings.warn("No comparison time added for SpeciesDeathRule in param dictionary. Defaulting to >.")
			self.comp = 1

	def get_species_and_parameters(self, dict fields):
		if "noise" in fields:
			return ([fields["specie"]], [fields["threshold"], fields["noise"]])
		else:
			return ([fields["specie"]], [fields["threshold"]])

#A death rule where death occurs when some parameter is greater than or less than a given threshold
cdef class ParamDeathRule(DeathRule):
	cdef unsigned param_ind
	cdef unsigned threshold_ind
	cdef unsigned threshold_noise_ind
	cdef unsigned has_noise
	cdef int comp
	cdef double threshold

	cdef int check_dead(self, double* state, double *params, double time, double volume, double initial_time, double initial_volume):
		self.threshold = params[self.threshold_ind]
		if self.has_noise > 0:
			self.threshold = self.threshold + normal_rv(0, params[self.threshold_noise_ind])

		if params[self.param_ind] > self.threshold - 1E-9  and params[self.param_ind] < self.threshold + 1E-9 and self.comp == 0:
			return 1
		elif params[self.param_ind] > self.threshold - 1E-9 and self.comp == 1:
			return 1
		elif params[self.param_ind] < self.threshold + 1E-9 and self.comp == -1:
			return 1
		else:
			return 0

	def initialize(self, dict param_dictionary, dict species_indices, dict parameter_indices):
		for (k, v) in param_dictionary.items():
			if k == "param":
				self.param_ind = parameter_indices[v]
			elif k == "threshold":
				self.threshold_ind = parameter_indices[v]
			elif k == "noise":
				self.threshold_noise_ind = parameter_indices[v]
				self.has_noise = 1
			elif k == "comp":
				if v == "=" or v =="equal":
					self.comp = 0
				elif v == "<" or v == "less":
					self.comp = -1
				elif v == ">" or v == "greater":
					self.comp = 1
			else:
				warnings.warn("Useless paramter for ParamDeathRule: "+str(k))
		if "noise" not in param_dictionary.keys():
			self.has_noise = 0
		if "comp" not in param_dictionary.keys():
			warnings.warn("No comparison time added for ParamDeathRule in param dictionary. Defaulting to >.")
			self.comp = 1

	def get_species_and_parameters(self, dict fields):
		if "noise" in fields:
			return ([], [fields["param"], fields["threshold"], fields["noise"]])
		else:
			return ([], [fields["param"], fields["threshold"]])

#A general death rule. Returns 1 if equation > 0. 0 otherwise
cdef class GeneralDeathRule(DeathRule):
	cdef Term equation
	cdef int check_dead(self, double* state, double *params, double time, double volume, double initial_time, double initial_volume):
		if (<Term> self.equation).volume_evaluate(state,params,volume,time) > 0:
			return 1
		else:
			return 0

	def initialize(self, dict param_dictionary, dict species_indices, dict parameter_indices):
		for (k, v) in param_dictionary.items():
			if k == "equation":
				self.equation = parse_expression(v, species_indices, parameter_indices)
			else:
				warnings.warn("Useless paramter for GeneralDivisionRule: "+str(k))

	def get_species_and_parameters(self, dict fields):
		equation_string = fields['equation'].strip()
		species, parameters = sympy_species_and_parameters(equation_string)
		return (species, parameters)

#A super class of Bioscrape.Types.Model which contains new cell lineage features
cdef class LineageModel(Model):
	cdef unsigned num_division_events
	cdef unsigned num_division_rules
	cdef unsigned num_death_events
	cdef unsigned num_death_rules
	cdef unsigned num_volume_events
	cdef unsigned num_volume_rules

	cdef list volume_events_list
	cdef list division_events_list
	cdef list division_rules_list
	cdef list death_events_list

	cdef vector[void*] c_lineage_propensities
	cdef list lineage_propensities
	cdef vector[void*] c_death_events
	cdef list death_events
	cdef vector[void*] c_division_events
	cdef list division_events
	cdef vector[void*] c_volume_events
	cdef list volume_events
	cdef vector[void*] c_other_events
	cdef list other_events
	cdef vector[void*] c_death_rules
	cdef list death_rules
	cdef vector[void*] c_division_rules
	cdef list division_rules
	cdef vector[void*] c_volume_rules
	cdef list volume_rules
	cdef list rule_volume_splitters
	cdef list event_volume_splitters

	cdef list global_species
	cdef double global_volume

	def __init__(self, filename = None, species = [], reactions = [], parameters = [], rules = [], events = [], global_species = [], global_volume = None, sbml_filename = None, initial_condition_dict = None, input_printout = False, initialize_model = True):


		self.volume_events_list = []
		self.volume_events = []
		self.division_events_list = []
		self.division_events = []
		self.death_events_list = []
		self.death_events = []
		self.volume_rules = []
		self.death_rules = []
		self.division_rules = []
		self.division_rules_list = []
		self.num_volume_rules = 0
		self.num_death_rules = 0
		self.num_division_rules = 0
		self.num_volume_events = 0
		self.num_death_events = 0
		self.num_division_events = 0

		#Filter out new rule types before calling super
		original_rules = []
		for rule in rules:
			if len(rule) == 2:
				rule_type, rule_attributes = rule
			elif len(rule) == 3:
				rule_type, rule_attributes, rule_frequency = rule

			if not ("Volume" in rule_type or "volume" in rule_type or 
				"Death" in rule_type or "death" in rule_type or 
				"Division" in rule_type or "division" in rule_type):
				original_rules.append(rule)

		if len(global_species) > 0 and global_volume == None:
			warnings.warn("global species added to LineageModel without the global_volume keyword being passed in. Global volume will vary dynamically and be equal to the total volume of all the cells.")
		elif len(global_species) == 0 and global_volume != None:
			warnings.warn("Setting global_volume without passing in the global_species keyword to LineageModel will do nothing unless you manually add global rections with the LineageModel.create_global_reaction function.")
		
		#Seperate reactions with global species inputs
		local_reactions = []
		global_reactions = []
		self.global_species = global_species
		if len(global_species) > 0:
			global_rxn_count = 0
			for rxn in reactions:
				reactants = rxn[0]
				if len(rxn) > 4:
					delay_reactants = rxn[5]
				else:
					delay_reactants = []
				if len([r for r in global_species if r in reactants]) > 0:
					global_reactions.append(rxn)
				else:
					local_reactions.append(rxn)
		else:
			local_reactions = reactions

		#Call super constructor
		super().__init__(filename = filename, species = species, reactions = local_reactions, parameters = parameters, rules = original_rules, initial_condition_dict = initial_condition_dict, sbml_filename = sbml_filename,  input_printout = input_printout, initialize_model = False)

		if global_volume == None:
			self.global_volume = 0
		else:
			self.global_volume = global_volume
		self._add_param("global_volume")
		self.set_parameter("global_volume", self.global_volume)

		#add global reactions
		global_rxn_count = 0
		for rxn in global_reactions:
			self.create_global_reaction(rxn, volume_param = "global_volume", volume_value = self.global_volume, identifier = global_rxn_count, global_species = self.global_species)
			global_rxn_count += 1

		#Add new types to the model
		for rule in rules:
			if len(rule) == 2:
				rule_type, rule_attributes = rule
			elif len(rule) == 3:
				rule_type, rule_attributes, rule_frequency = rule

			if "Volume" in rule_type or "volume" in rule_type:
				self.create_volume_rule(rule_type, rule_attributes)
			elif "Death" in rule_type or "death" in rule_type:
				self.create_death_rule(rule_type, rule_attributes)
			elif "Division" in rule_type or "division" in rule_type:
				self.create_division_rule(rule_type, rule_attributes)

		for event in events:
			if len(event) == 4:
				event_type, event_params, event_propensity, propensity_params = event
			else:
				raise ValueError("Events must be tuples: (event_type (str), event_params (dict), event_propensity (str), propensity_params (dict)).")
			if "Volume" in event_type or "volume" in event_type:
				self.create_volume_event(event_type, event_params, event_propensity, propensity_params)
			elif "Death" in event_type or "death" in event_type:
				self.create_death_event(event_type, event_params, event_propensity, propensity_params)
			elif "Division" in event_type or "division" in event_type:
				self.create_division_event(event_type, event_params, event_propensity, propensity_params)
			else:
				raise ValueError("Unknown Event Type:", event_type)

		if initialize_model:
			self._initialize()
			

	def _create_vectors(self):
		#Create c-vectors of different objects
		super()._create_vectors()

		for rule_object in self.repeat_rules:
			self.c_repeat_rules.push_back(<void*> rule_object)

		self.num_volume_rules = len(self.volume_rules)
		for i in range(self.num_volume_rules):
			rule = self.volume_rules[i]
			self.c_volume_rules.push_back(<void*> rule)

		self.num_death_rules = len(self.death_rules)
		for i in range(self.num_death_rules):
			rule = self.death_rules[i]
			self.c_death_rules.push_back(<void *> rule)

		
		self.rule_volume_splitters = []
		self.event_volume_splitters = []
		self.num_division_rules = len(self.division_rules_list)
		for i in range(self.num_division_rules):
			rule, volume_splitter = self.division_rules_list[i]
			self.division_rules.append(rule)
			self.c_division_rules.push_back(<void*>rule)
			self.rule_volume_splitters.append(volume_splitter)

		#Propensity Order:
		# Reactionns, Divison Events, Volume Events, Death Events
		self.lineage_propensities = []

		self.num_division_events = len(self.division_events_list)
		for i in range(self.num_division_events):
			event, prop_object, volume_splitter = self.division_events_list[i]
			self.division_events.append(event)
			self.c_division_events.push_back(<void*> event)
			self.lineage_propensities.append(prop_object)
			self.c_lineage_propensities.push_back(<void*>prop_object)
			self.event_volume_splitters.append(volume_splitter)
	
		self.num_volume_events = len(self.volume_events_list)
		for i in range(self.num_volume_events):
			event, prop_object = self.volume_events_list[i]
			self.lineage_propensities.append(prop_object)
			self.c_lineage_propensities.push_back(<void*>prop_object)
			self.volume_events.append(event)
			self.c_volume_events.push_back(<void*>event)

		self.num_death_events = len(self.death_events_list)
		for i in range(self.num_death_events):
			event, prop_object = self.death_events_list[i]
			self.lineage_propensities.append(prop_object)
			self.c_lineage_propensities.push_back(<void*>prop_object)
			self.death_events.append(event)
			self.c_death_events.push_back(<void*>event)
		

	def py_initialize(self):
		self._initialize()
		self.initialized = True

	def py_get_event_counts(self):
		return self.num_division_events, self.num_volume_events, self.num_death_events
	def py_get_rule_counts(self):
		return self.num_division_rules, self.num_volume_rules, self.num_death_rules

	def add_event(self, Event event_object, dict event_param_dict, Propensity prop_object, dict propensity_param_dict, str event_type = None, VolumeSplitter volume_splitter = None):
		self.initialized = False

		species_names_e, param_names_e = event_object.get_species_and_parameters(event_param_dict)
		species_names_p, param_names_p = prop_object.get_species_and_parameters(propensity_param_dict)

		for species_name in species_names_e+species_names_p:
			self._add_species(species_name)
		for param_name in param_names_e+param_names_p:
			self._add_param(param_name)

		event_object.initialize(event_param_dict, self.species2index, self.params2index)
		prop_object.initialize(propensity_param_dict, self.species2index, self.params2index)

		if event_type in ["division", "Division", "division event", "DivisionEvent", "Division Event"]:
			if volume_splitter == None:
				raise ValueError("DivisionRules require a VolumeSplitter Object to be passed into add_event")
			self.division_events_list.append((event_object, prop_object, volume_splitter))

		elif event_type in ["volume", "Volume", "volume event", "VolumeEvent", "Volume Event"]:
			self.volume_events_list.append((event_object, prop_object))
		elif event_type in ["death", "Death", "death event", "DeathEvent", "Death Event"]:
			self.death_events_list.append((event_object, prop_object))
		else:
			raise ValueError("Unknown Event Type: Misc Event Not Yet Implemented")
			self.other_events_list.append((event_object, prop_object))

	def create_death_event(self, str event_type, dict event_params, str event_propensity_type, dict propensity_params, print_out = False):
		event_params = dict(event_params)
		propensity_params = dict(propensity_params)
		prop_object = self.create_propensity(event_propensity_type, propensity_params, print_out = print_out)
		if event_type in ["", "death", "DeathEvent", "death event", "Death Event", "default", "Default"]:
			event_object = DeathEvent()
		else:
			raise ValueError("Unknwown DeathEvent type"+str(event_type))
		self.add_event(event_object, event_params, prop_object, propensity_params, event_type = "death")

	def create_division_event(self, str event_type, dict event_params, str event_propensity_type, dict propensity_params, VolumeSplitter volume_splitter, print_out = False):
		if print_out:
			print("Adding New DivisionEvent with event_type=", event_type, "params=", event_params, "propensity_type=",event_propensity_type, "propensity_params=", propensity_params, "and VolumeSplitter=", volume_splitter)
		event_params = dict(event_params)
		propensity_params = dict(propensity_params)
		prop_object = self.create_propensity(event_propensity_type, propensity_params, print_out = print_out)
		if event_type in ["", "division", "Division", "DivisionEvent", "division event", "Division Event", "deafult"]:
			event_object = DivisionEvent()
		else:
			raise ValueError("Unknown DivisionEvent type"+str(event_type))


		self.add_event(event_object, event_params, prop_object, propensity_params, event_type = "division", volume_splitter = volume_splitter)

	def create_volume_event(self, event_type, dict event_params, str event_propensity_type, dict propensity_params, print_out = False):
		event_params = dict(event_params)
		propensity_params = dict(propensity_params)
		if print_out:
			warnings.warn("Creating New Volume event\n\ttype="+event_type+"\n\tparams="+str(event_params)+"\n\tprop_type="+str(event_propensity_type)+"\n\tprop_params="+str(propensity_params))
		prop_object = self.create_propensity(event_propensity_type, propensity_params, print_out = print_out)

		if event_type in ["linear", "Linear", "Linear Volume", "linear volume", "LinearVolume" "LinearVolumeEvent", "linear volume event", "Linear Volume Event"]:
			self._param_dict_check(event_params, "growth_rate", "DummyVar_LinearVolumeEvent")
			event_object = LinearVolumeEvent()
		elif event_type in ["multiplicative", "multiplicative", "Multiplicative Volume", "multiplicative volume", "MultiplicativeVolume", "MultiplicativeVolumeEvent", "Multiplicative Volume Event", "multiplicative volume event"]:
			self._param_dict_check(event_params, "growth_rate", "DummyVar_MultiplicativeVolumeEvent")
			event_object = MultiplicativeVolumeEvent()
		elif event_type in ["general", "General", "General Volume", "general volume", "GeneralVolume", "GeneralVolumeEvent", "General Volume Event", "general volume event"]:
			event_object = GeneralVolumeEvent()
		else:
			raise ValueError("Unknown VolumeEvent Type: "+str(event_type))

		self.add_event(event_object, event_params, prop_object, propensity_params, event_type = "volume")

	def add_lineage_rule(self, LineageRule rule_object, dict rule_param_dict, str rule_type, VolumeSplitter volume_splitter = None):
		species_names, param_names = rule_object.get_species_and_parameters(rule_param_dict)

		for species_name in species_names:
			self._add_species(species_name)
		for param_name in param_names:
			self._add_param(param_name)
		if "division" in rule_type or "Division" in rule_type:
			if volume_splitter == None:
				raise ValueError("DivisionRules must be added with a volume splitter object in add_lineage_rule.")
			rule_object.initialize(rule_param_dict, self.species2index,  self.params2index)
			self.division_rules_list.append((rule_object, volume_splitter))
		else:
			rule_object.initialize(rule_param_dict, self.species2index,  self.params2index)
			if "death" in rule_type or "Death" in rule_type:
				self.death_rules.append(rule_object)
			elif "volume" in rule_type or "Volume" in rule_type:
				self.volume_rules.append(rule_object)
			else:
				raise ValueError("add_lineage_rule only takes rules of type 'DeathRule', 'DivisionRule', and 'VolumeRule'. For Other rule types, consider trying Model.add_rule.")

	def create_death_rule(self, str rule_type, dict rule_param_dict):
		if rule_type in ["species", "Species", "SpeciesDeathRule"]:
			self._param_dict_check(rule_param_dict, "specie", "DummyVar_SpeciesDeathRule")
			self._param_dict_check(rule_param_dict, "threshold", "DummyVar_SpeciesDeathRule")
			if "noise" in rule_param_dict:
				self._param_dict_check(rule_param_dict, "noise", "DummyVar_SpeciesDeathRule")
			rule_object = SpeciesDeathRule()
		elif rule_type in ["param", "parameter", "Param", "Parameter", "ParamDeathRule"]:
			self._param_dict_check(rule_param_dict, "param", "DummyVar_ParamDeathRule")
			self._param_dict_check(rule_param_dict, "threshold", "DummyVar_ParamDeathRule")
			if "noise" in rule_param_dict:
				self._param_dict_check(rule_param_dict, "noise", "DummyVar_ParamDeathRule")
			rule_object = ParamDeathRule()
		elif rule_type in ["general", "General", "GeneralDeathRule"]:
			rule_object = GeneralDeathRule()
		else:
			raise ValueError("Unknown DeathRule type: "+str(rule_type))

		self.add_lineage_rule(rule_object, rule_param_dict, rule_type = "death")

	def create_division_rule(self, str rule_type, dict rule_param_dict, VolumeSplitter volume_splitter):
		if rule_type in ["time", "Time", "TimeDivisionRule"]:
			self._param_dict_check(rule_param_dict, "threshold", "DummyVar_TimeDeathRule")
			if "noise" in rule_param_dict:
				self._param_dict_check(rule_param_dict, "noise", "DummyVar_TimeDeathRule")
			rule_object = TimeDivisionRule()
		elif rule_type in ["volume", "Volume", "VolumeDivisionRule"]:
			self._param_dict_check(rule_param_dict, "threshold", "DummyVar_VolumeDeathRule")
			if "noise" in rule_param_dict:
				self._param_dict_check(rule_param_dict, "noise", "DummyVar_VolumeDeathRule")
			rule_object = VolumeDivisionRule()
		elif rule_type in ["delta", "Delta", "deltaV", "DeltaV", "DeltaVDivisionRule"]:
			self._param_dict_check(rule_param_dict, "threshold", "DummyVar_DeltaVDeathRule")
			if "noise" in rule_param_dict:
				self._param_dict_check(rule_param_dict, "noise", "DummyVar_DeltaVDeathRule")
			rule_object = DeltaVDivisionRule()
		elif rule_type in ["general", "General", "GeneralDivisionRule"]:
			rule_object = GeneralDivisionRule()
		else:
			raise ValueError("Unknown DivisionRule type: "+str(rule_type))
		self.add_lineage_rule(rule_object, rule_param_dict, rule_type = 'division', volume_splitter = volume_splitter)

	def create_volume_rule(self, str rule_type, dict rule_param_dict):
		if rule_type in ["linear", "Linear", "LinearVolumeRule"]:
			self._param_dict_check(rule_param_dict, "growth_rate", "DummyVar_LinearVolumeRule")
			if "noise" in rule_param_dict:
				self._param_dict_check(rule_param_dict, "noise", "DummyVar_LinearVolumeRule")
			rule_object = LinearVolumeRule()
		elif rule_type in ["multiplicative", "MultiplicativeVolume", "MultiplicativeVolumeRule"]:
			self._param_dict_check(rule_param_dict, "growth_rate", "DummyVar_MultiplicativeVolumeRule")
			if "noise" in rule_param_dict:
				self._param_dict_check(rule_param_dict, "noise", "DummyVar_MultiplicativeVolumeRule")
			rule_object = MultiplicativeVolumeRule()
		elif rule_type in ["assignment", "Assignment", "AssignmentVolumeRule"]:
			rule_object = AssignmentVolumeRule()
		elif rule_type in ["ode", "ODE", "ODEVolumeRule"]:
			rule_object = ODEVolumeRule()
		else:
			raise ValueError("Unknown VolumeRule type: "+str(rule_type))
		self.add_lineage_rule(rule_object, rule_param_dict, rule_type = 'volume')

	
	def create_global_reaction(self, rxn, volume_param = "global_volume", volume_value = 1, identifier = ""):

		raise NotImplementedError("Do I even want global reactions like this? I think not!")

		if len(rxn) == 4:
			reactants, products, propensity_type, propensity_param_dict = rxn
			delay_type, delay_reactants, delay_products, delay_param_dict = None, None,  None, None
		elif len(rxn) == 8:
			reactants, products, propensity_type, propensity_param_dict, delay_type, delay_reactants, delay_products, delay_param_dict = rxn
		else:
			raise ValueError("Reaction Tuple of the wrong length! Must be of length 4 (no delay) or 8 (with delays). See BioSCRAPE Model API for details.")
		

		if False in [p in self.global_species for p in products] or False in [r in self.global_species for r in reactants]:
			raise ValueError(f"Global Reaction {reactants} --> {products} contains non-global species.")

		elif len(global_species) == 0:
			warnings.warn("No global species defined for this model or passed into create_global_reaction. Defaulting to non-global reaction.")
			self.create_reaction(reactants, products, propensity_type, propensity_param_dict, delay_type, delay_reactants, delay_products, delay_param_dict)
		elif "k" not in propensity_param_dict:
			warnings.warn("create_global_reaction only works with propensities that have a rate parameter 'k' in their param_dictionary. propensity_type="+propensity_type+" either doesn't have the proper parameter or is incompatible with automatic global reactions. This reaction will be added but not rescaled.") 
			self.create_reaction(reactants, products, propensity_type, propensity_param_dict, delay_type, delay_reactants, delay_products, delay_param_dict)
		else:
			old_val = propensity_param_dict["k"]
			try:
				float(old_val)
				float_val = True
			except ValueError:
				float_val = False

			rate_var = "global_reaction_"+propensity_type+"_k_rescaled_"+str(identifier)
			n_global = len([r for r in reactants if r in global_species])
			
			self._add_param(rate_var)
			if float_val:
				rule_equation = "_"+rate_var + "=" + str(old_val)+"/"+"(_"+volume_param+"^"+str(n_global)+")"
				self.set_parameter(rate_var, 1.*old_val/(volume_value**n_global))
			else:
				rule_equation = "_"+rate_var + "= _"+old_val+"/"+"(_"+volume_param+"^"+str(n_global)+")"
				self._add_param(old_val)
				self.set_parameter(rate_var, 1./(volume_value**n_global))
			propensity_param_dict["k"] = rate_var
			

			self._add_param(volume_param)
			self.set_parameter(volume_param, volume_value)
			self.create_reaction(reactants, products, propensity_type, propensity_param_dict, delay_type, delay_reactants, delay_products, delay_param_dict)
			self.create_rule("assignment", {"equation":rule_equation})





	cdef unsigned get_num_division_rules(self):
		return self.num_division_rules
	cdef unsigned get_num_volume_rules(self):
		return self.num_volume_rules
	cdef unsigned get_num_death_rules(self):
		return self.num_death_rules
	cdef unsigned get_num_division_events(self):
		return self.num_division_events
	cdef unsigned get_num_volume_events(self):
		return self.num_volume_events
	cdef unsigned get_num_death_events(self):
		return self.num_death_events
	cdef list get_lineage_propensities(self):
		return self.lineage_propensities
	def py_get_lineage_propensities(self):
		return self.lineage_propensities
	def py_get_num_lineage_propensities(self):
		return len(self.lineage_propensities)
	def py_get_num_division_rules(self):
		return self.get_num_division_rules()
	def py_get_num_volume_rules(self):
		return self.get_num_volume_rules()
	def py_get_num_death_rules(self):
		return self.get_num_death_rules()
	def py_get_num_division_events(self):
		return self.get_num_division_events()
	def py_get_num_volume_events(self):
		return self.get_num_volume_events()
	def py_get_num_death_events(self):
		return self.get_num_death_events()

	cdef (vector[void*])* get_c_lineage_propensities(self):
		return & self.c_lineage_propensities
	cdef (vector[void*])* get_c_division_rules(self):
		return & self.c_division_rules
	cdef (vector[void*])* get_c_volume_rules(self):
		return & self.c_volume_rules
	cdef (vector[void*])* get_c_death_rules(self):
		return & self.c_death_rules
	cdef (vector[void*])* get_c_division_events(self):
		return & self.c_division_events
	cdef (vector[void*])* get_c_volume_events(self):
		return & self.c_volume_events
	cdef (vector[void*])* get_c_death_events(self):
		return & self.c_death_events

	def py_get_volume_splitters(self):
		return self.rule_volume_splitters, self.event_volume_splitters

cdef class LineageCSimInterface(ModelCSimInterface):
	cdef unsigned num_division_events
	cdef unsigned num_division_rules
	cdef unsigned num_death_events
	cdef unsigned num_death_rules
	cdef unsigned num_volume_events
	cdef unsigned num_volume_rules
	cdef unsigned num_lineage_propensities

	cdef list volume_events_list
	cdef list division_events_list
	cdef list death_events_list	

	cdef vector[void*] *c_lineage_propensities
	cdef vector[void*] *c_death_events
	cdef vector[void*] *c_division_events
	cdef vector[void*] *c_volume_events
	cdef vector[void*] *c_other_events
	cdef vector[void*] *c_death_rules
	cdef vector[void*] *c_division_rules
	cdef vector[void*] *c_volume_rules

	cdef list division_event_volume_splitters
	cdef list division_rule_volume_splitters

	def __init__(self, LineageModel M):
		super().__init__(M)
		self.num_division_rules = <unsigned>M.py_get_num_division_rules()
		self.num_volume_rules = <unsigned>M.py_get_num_volume_rules()
		self.num_death_rules = <unsigned>M.py_get_num_death_rules()
		self.num_division_events = <unsigned>M.py_get_num_division_events()
		self.num_volume_events = <unsigned>M.py_get_num_volume_events()
		self.num_death_events = <unsigned>M.py_get_num_death_events()
		self.num_lineage_propensities = <unsigned>M.py_get_num_lineage_propensities()
		self.c_lineage_propensities = M.get_c_lineage_propensities()
		self.c_volume_rules = M.get_c_volume_rules()
		self.c_division_rules = M.get_c_division_rules()
		self.c_death_rules = M.get_c_death_rules()
		self.c_volume_events = M.get_c_volume_events()
		self.c_division_events = M.get_c_division_events()
		self.c_death_events = M.get_c_death_events()
		self.division_rule_volume_splitters, self.division_event_volume_splitters = M.py_get_volume_splitters()


	#similar to compute_stochastic_volume_propensities but events are new included as well
	cdef void compute_lineage_propensities(self, double[:] state, double[:] propensity_destination, double volume, double time):
		cdef unsigned ind
		for ind in range(self.num_reactions):
			propensity_destination[ind] = (<Propensity>(self.c_propensities[0][ind])).get_stochastic_volume_propensity(&state[0], self.c_param_values, volume, time)

		for ind in range(self.num_lineage_propensities):
			propensity_destination[self.num_reactions+ind] = (<Propensity>(self.c_lineage_propensities[0][ind])).get_stochastic_volume_propensity(&state[0], self.c_param_values, volume, time)

	#Applies all rules that change the volume of a cell
	cdef double apply_volume_rules(self, double* state, double volume, double time, double dt):
		cdef int ind
		for ind in range(self.num_volume_rules):
			volume = (<VolumeRule>self.c_volume_rules[0][ind]).get_volume(state, self.c_param_values, volume, time, dt)
		return volume

	#Applies death rules in the order they were added to the model. Returns the index of the first death rule that returns True. -1 otherwise.
	cdef int apply_death_rules(self, double* state, double volume, double time, double start_volume, double start_time):
		cdef int isdead = 0
		cdef int ind
		for ind in range(self.num_death_rules):
			isdead = (<DeathRule>self.c_death_rules[0][ind]).check_dead(state, self.c_param_values, time, volume, start_time, start_volume)
			if isdead > 0:
				return ind
		return -1

	#Applies divison rules in the order they were added to the model. Returns the index of the first division rule that returns True. -1 otherwise
	cdef int apply_division_rules(self, double* state, double volume, double time, double start_volume, double start_time):
		cdef int divided = 0
		cdef int ind
		for ind in range(self.num_division_rules):
			divided = (<DivisionRule>self.c_division_rules[0][ind]).check_divide(state, self.c_param_values, time, volume, start_time, start_volume)
			if divided > 0:
				return ind
		return -1

	#Applies a single volume event, determined by the index passed in
	cdef double apply_volume_event(self, int event_index, double* state, double current_time, double current_volume):
		current_volume = (<VolumeEvent>self.c_volume_events[0][event_index]).get_volume(state, self.c_param_values, current_volume, current_time)
		return current_volume

	#Divides a single cell using a VolumeSplitter determined by vsplit_index
	cdef np.ndarray partition(self, int vsplit_ind, LineageVolumeCellState parent):
		cdef VolumeSplitter vsplit
		if vsplit_ind >= self.num_division_rules and vsplit_ind<self.num_division_rules+self.num_division_events:
			vsplit_ind = vsplit_ind - self.num_division_rules
			vsplit = self.division_event_volume_splitters[vsplit_ind]
		elif vsplit_ind < self.num_division_rules and vsplit_ind >= 0:
			vsplit = self.division_rule_volume_splitters[vsplit_ind]
		else:
			raise ValueError('Invalid volume splitter index: vsplit_ind='+str(vsplit_ind))

		return vsplit.partition(parent)


	cdef np.ndarray delay_partition(self, unsigned vsplit_ind, LineageVolumeCellState parent):
		raise NotImplementedError("Implement me!")

	cdef unsigned get_num_lineage_propensities(self):
		return self.num_lineage_propensities

	cdef unsigned get_num_division_events(self):
		return self.num_division_events
	cdef unsigned get_num_volume_events(self):
		return self.num_volume_events
	cdef unsigned get_num_death_events(self):
		return self.num_death_events
	cdef unsigned get_num_volume_rules(self):
		return self.num_volume_rules
	cdef unsigned get_num_death_rules(self):
		return self.num_death_rules
	cdef unsigned get_num_division_rules(self):
		return self.num_division_rules

#A new wrapper for the VolumeCellState with new internal variables
cdef class LineageVolumeCellState(DelayVolumeCellState):
	cdef double initial_volume #Stores the birth Volume
	cdef double initial_time #Stores the time the Cell was "born"
	#divided = -1: Cell Not divided
	#divided E [0, num_division_rules): DivisionRule divided caused the cell to divide
	#divided E [num_division_rules, num_division_rules + num_division_events]: Division Event divided-num_division_rules caused the cell to divide
	cdef int divided
	#dead = -1: Cell Not dead
	#dead E [0, num_death_rules): DeathRule divided caused the cell to die
	#dead E [num_death_rules, num_death_rules + num_death_events]: DeathEvent dead-num_death_rules caused the cell to die 
	cdef int dead
	cdef state_set

	def __init__(self, v0 = 0, t0 = 0, state = []):
		self.set_initial_vars(v0, t0)
		self.set_volume(v0)
		self.set_time(t0)
		self.volume_object = None
		self.divided = -1
		self.dead = -1
		if len(state) == 0:
			self.state_set = 0
		else:
			self.state_set = 1
			self.py_set_state(state)

	def get_state_set(self):
		return self.state_set

	def py_set_state(self, state):
		self.state_set = 1
		return super().py_set_state(np.asarray(state))

	cdef void set_initial_vars(self, double volume, double time):
		self.initial_volume = volume
		self.initial_time = time

	cdef double get_initial_volume(self):
		return self.initial_volume

	cdef void set_state_comp(self, double val, unsigned comp_ind):
		self.state[comp_ind] = val

	cdef double get_state_comp(self, unsigned comp_ind):
		return self.state[comp_ind]

	def py_get_initial_volume(self):
		return self.get_initial_volume()

	cdef double get_initial_time(self):
		return self.initial_time

	def py_get_initial_time(self):
		return self.get_initial_time()

	cdef void set_divided(self, divided):
		self.divided = divided

	cdef int get_divided(self):
		return self.divided

	cdef void set_dead(self, dead):
		self.dead = dead
	cdef int get_dead(self):
		return self.dead

cdef class SingleCellSSAResult(VolumeSSAResult):
	#divided = -1: Cell Not divided
	#divided E [0, num_division_rules): DivisionRule divided caused the cell to divide
	#divided E [num_division_rules, num_division_rules + num_division_events]: Division Event divided-num_division_rules caused the cell to divide
	cdef int divided
	#dead = -1: Cell Not dead
	#dead E [0, num_death_rules): DeathRule divided caused the cell to die
	#dead E [num_death_rules, num_death_rules + num_death_events]: DeathEvent dead-num_death_rules caused the cell to die 
	cdef int dead
	
	cdef void set_dead(self, int dead):
		self.dead = dead
	def py_set_dead(self, dead):
		self.set_dead(dead)

	cdef int get_dead(self):
		return self.dead
	def py_get_dead(self):
		return self.get_dead()

	cdef void set_divided(self, int divided):
		self.divided = divided
	def py_set_divided(self, divided):
		self.set_divided(divided)

	cdef int get_divided(self):
		return self.divided

	def py_get_divided(self):
		return self.get_divided()

	cdef VolumeCellState get_final_cell_state(self):
		cdef unsigned final_index = (<np.ndarray[np.double_t,ndim=1]> self.timepoints).shape[0]-1
		cdef LineageVolumeCellState cs  = LineageVolumeCellState(t0 = self.timepoints[0], v0 = self.volume[0], state = self.simulation_result[final_index,:])
		cs.set_time(self.timepoints[final_index])
		cs.set_volume(self.volume[final_index])
		cs.set_divided(self.divided)
		cs.set_dead(self.dead)
		return cs


cdef class LineageVolumeSplitter(VolumeSplitter):
	cdef unsigned how_to_split_v
	cdef vector[int] binomial_indices
	cdef vector[int] perfect_indices
	cdef vector[int] duplicate_indices
	cdef vector[int] custom_indices
	cdef double partition_noise
	cdef dict ind2customsplitter
	cdef dict custom_partition_functions

	def __init__(self, Model M, options = {}, custom_partition_functions = {}, partition_noise = .5):
		self.ind2customsplitter == {}
		self.custom_partition_functions = custom_partition_functions
		if self.partition_noise > 1:
			raise ValueError("Partition Noise must be between 0 and 1")
		self.partition_noise = partition_noise

		#Figure out how volume will be split
		if "volume" not in options or options["volume"] == "binomial":
			self.how_to_split_v = 0
		elif options["volume"] == "duplicate":
			self.how_to_split_v = 1
		elif options["volume"] == "perfect":
			self.how_to_split_v = 2
		elif options["volume"] in custom_partition_functions:
			self.how_to_split_v = 3
			self.ind2customsplitter["volume"] = options["volume"]
		else:
			raise ValueError("Custom partition function key, "+str(options["volume"])+", for 'volume' not in custom_partition_functions")

		#Figure out how other species are split
		for s in M.get_species2index():
			index = M.get_species_index(s)
			if s not in options or options[s] == "binomial":
				self.binomial_indices.push_back(index)
			elif options[s] == "duplicate":
				self.duplicate_indices.push_back(index)
			elif options[s] == "perfect":
				self.perfect_indices.push_back(index)
			elif options[s] in custom_partition_functions:
				self.custom_indices.push_back(index)
				self.ind2customsplitter[index] = options[s]
			else:
				raise ValueError("Custom partition function key, "+str(options["volume"])+", for "+s+" not in custom_partition_functions")

	cdef np.ndarray partition(self, VolumeCellState parent):
		cdef double v0d, v0e, t0, p, q, c1, c2
		# set times
		t0 = parent.get_time()

		# partition the states, copying already takes care of duplication replications.
		cdef np.ndarray[np.double_t,ndim=1] dstate = parent.get_state().copy()
		cdef np.ndarray[np.double_t,ndim=1] estate = parent.get_state().copy()
		cdef unsigned length = dstate.shape[0]

		cdef unsigned loop_index = 0
		cdef unsigned species_index = 0
		cdef unsigned amount = 0
		cdef unsigned amount2 = 0
		cdef double d_value = 0.0
		
		# simulate partitioning noise
		if self.how_to_split_v == 0: #Binomial
			p = 0.5 - cyrandom.uniform_rv()*self.partition_noise/2.
			q = 1 - p
			# calculate binomial volumes
			v0d = parent.get_volume() * p
			v0e = parent.get_volume() * q
		elif self.how_to_split_v == 1: #Duplicate
			p = 1
			q = 1
			v0d = parent.get_volume()
			v0e = parent.get_volume()
		elif self.how_to_split_v == 2: #perfect
			p = .5
			q = .5
			v0d = parent.get_volume()*.5
			v0e = parent.get_volume()*.5
		else:
			splitter = self.custom_partition_functions[self.ind2customsplitter["volume"]]
			v0d, v0e = splitter(parent, "volume")
			p = v0d/parent.get_volume()
			q = v0e/parent.get_volume()
			if v0d <= 0 or v0e <= 0:
				raise ValueError("splitter "+self.ind2customsplitter[species_index]+" returned negative quantities for volume")
 
		# take care of perfect splitting
		for loop_index in range(self.perfect_indices.size()):
			species_index = self.perfect_indices[loop_index]
			d_value = p * dstate[species_index]
			amount = <int> (d_value+0.5)
			if d_value-amount <= 1E-8 and amount>0:
				dstate[species_index] = <double> amount
			elif amount <0:
				raise ValueError('negative quantity in perfect partitioning')
			else:
				if cyrandom.uniform_rv() <= p:
					dstate[species_index] = <int> d_value + 1
				else:
					dstate[species_index] = <int> d_value
			estate[species_index] -= dstate[species_index]

		# take care of binomial splitting
		for loop_index in range(self.binomial_indices.size()):
			species_index = self.binomial_indices[loop_index]
			amount = cyrandom.binom_rnd_f(dstate[species_index],p)
			dstate[species_index] = <double> amount
			estate[species_index] -= dstate[species_index]

		for loop_index in range(self.custom_indices.size()):
			species_index = self.custom_indices[loop_index]
			splitter = self.custom_partition_functions[self.ind2customsplitter[species_index]]
			c1, c2 = splitter(species_index, parent)
			if c1 < 0 or c2 < 0:
				raise ValueError("splitter "+self.ind2customsplitter[species_index]+" returned negative quantities for species index "+str(species_index))
			dstate[species_index] = <double> c1
			estate[species_index] = <double> c2   

		# create return structure
		cdef np.ndarray ans = np.empty(2, dtype=np.object)
		cdef LineageVolumeCellState d = LineageVolumeCellState(v0 = v0d, t0 = parent.get_time(), state = dstate)
		cdef LineageVolumeCellState e = LineageVolumeCellState(v0 = v0e, t0 = parent.get_time(), state = estate)
		ans[0] = d
		ans[1] = e
		return ans 

cdef class LineageSSASimulator:

	#Memory Views are reused between each individual cell
	#Sometimes, to be compatable with the double* used in original bioscrape, these are cast to double*
	#these are used by SimulateSingleCell
	cdef double[:] c_timepoints, c_current_state, c_propensity, c_truncated_timepoints, c_volume_trace
	cdef double[:, :] c_stoich, c_results
	
	#An Interface is stored ina  linear model for fast helper functions
	cdef LineageCSimInterface interface

	#All the counters are also reused for each individual cell and set when interface is set
	cdef unsigned num_species, num_reactions, num_volume_events, num_death_events, num_division_events, num_volume_rules, num_death_rules, num_division_rules, num_propensities, num_timepoints

	#These are used by SimulateLineage and PropogateCells
	cdef Schnitz s, daughter_schnitz1, daughter_schnitz2
	cdef LineageVolumeCellState d1, d2, d1final, d2final, cs, dcs
	cdef SingleCellSSAResult r
	cdef list old_schnitzes, old_cell_states
	cdef Lineage lineage

	#Used for PropogateCells
	cdef list cell_states

	#Used to create a propensity buffer from an interface
	cdef create_propensity_buffer(self, LineageCSimInterface interface):
		cdef np.ndarray[np.double_t, ndim = 1] c_propensity = np.zeros(interface.get_num_lineage_propensities()+interface.get_num_reactions())
		return c_propensity

	#Python accessible version
	def py_create_propensity_buffer(self, LineageCSimInterface interface):
		return self.create_propensity_buffer(interface)

	#helper function to take an np.ndarrays and set them to internal memory views
	cdef void set_c_truncated_timepoints(self, np.ndarray timepoints):
		self.c_truncated_timepoints = timepoints
	#python accessible version
	def py_set_c_truncated_timepoints(self, np.ndarray timepoints):
		self.set_c_truncated_timepoints(timepoints)

	cdef void set_c_timepoints(self, np.ndarray timepoints):
		self.c_timepoints = timepoints
	#Python accessible version
	def py_set_c_timepoints(self, np.ndarray timepoints):
		self.set_c_timepoints(timepoints)

	#Sets the internal interface and associated internal variables
	#Main speed-up due to not having to set c_stoic (which could be very large) as often
	cdef void intialize_single_cell_interface(self, LineageCSimInterface interface):
		#Reset internal variables
		self.interface = interface

		#Memory View Setup
		#Stochiomettric Matrix
		self.c_stoich = self.interface.get_update_array() + self.interface.get_delay_update_array()
		#Prepare propensity buffer of the right size
		self.c_propensity = self.create_propensity_buffer(self.interface)

		self.num_species = self.interface.get_number_of_species()
		self.num_reactions = self.interface.get_number_of_species()

		self.num_volume_events = self.interface.get_num_volume_events()
		self.num_death_events = self.interface.get_num_death_events()
		self.num_division_events = self.interface.get_num_division_events()
		self.num_volume_rules = self.interface.get_num_volume_rules()
		self.num_death_rules = self.interface.get_num_death_rules()
		self.num_division_rules = self.interface.get_num_division_rules()
		self.num_propensities = self.num_reactions + self.num_volume_events + self.num_death_events + self.num_division_events
	#Python accessible version
	def py_initialize_single_cell_interface(self, LineageCSimInterface interface):
		self.initialize_single_cell_interface(interface)

	#SSA for a single cell. Simulates until it devides or dies using division / death rules and/or reactions.
	#Before calling this for a given interface, must call initialize_single_cell_interface to set up internal variables
	#Python wrapper below takes care of all the details, but will be slower if used repeatedly
	cdef SingleCellSSAResult SimulateSingleCell(self, LineageVolumeCellState v, double[:] timepoints):
		#print("SimulateSingleCell")

		#Memory views are reused from other objects for less allocation
		cdef unsigned num_timepoints = len(timepoints)

		cdef double initial_time = v.get_initial_time()
		cdef double current_time = v.get_time()
		cdef double final_time = timepoints[num_timepoints-1]
		cdef double proposed_time = 0.0
		cdef unsigned current_index = 0
		cdef unsigned reaction_choice = 4294967295 # https://en.wikipedia.org/wiki/4,294,967,295
		cdef unsigned species_index = 4294967295
		cdef double delta_t = timepoints[1]-timepoints[0]
		cdef double next_queue_time = timepoints[current_index+1]
		cdef double move_to_queued_time = 0
		cdef double initial_volume = v.get_initial_volume()
		cdef double current_volume = v.get_volume()
		cdef int cell_divided = -1
		cdef int cell_dead = -1
		cdef double Lambda = 0

		#Must initialize Memory Views
		self.c_truncated_timepoints = timepoints

		#These are kept as local numpy arrays because they are returned after every simulation
		cdef np.ndarray[np.double_t,ndim=2] results = np.zeros((num_timepoints,self.num_species))
		self.c_results = results
		cdef np.ndarray[np.double_t,ndim=1] volume_trace = np.zeros(num_timepoints,)
		self.c_volume_trace = volume_trace
		cdef SingleCellSSAResult SCR

		#Set Initial State
		if v.get_state_set() == 1:
			self.c_current_state = v.py_get_state().copy()
		else:
			warnings.warn("No initial state set (via LineageVolumeCellState v) in SingleCellSSAResuslt. Defaulting to the Model's initial state.")
			self.c_current_state = self.interface.get_initial_state().copy()
			v.py_set_state(self.c_current_state)

		#Warn user if delays are in the model (which will be converted to non-delay reactions)
		if (self.interface.py_get_delay_update_array() != np.zeros(self.interface.py_get_delay_update_array().shape)).any():
			warnings.warn("Delay reactions found in the model. SingleCellSSASimulator will simulate these reactions without delay. Delays are not yet supported for LineageModels but can be simulated as regular Models with the DelayVolumeSSASimulator.")

		# Do the SSA part now
		#print("SingleCell loop start")
		while current_index < num_timepoints:
			
			# Compute rules in place
			self.interface.apply_repeated_volume_rules(&self.c_current_state[0], current_volume, current_time)

			#returns the index of the first DeathRule that returned True and -1 otherwise
			cell_dead = self.interface.apply_death_rules(&self.c_current_state[0], current_volume, current_time, initial_volume, initial_time)

			#returns the index of the first DivisionRule that returned True and -1 otherwise
			cell_divided = self.interface.apply_division_rules(&self.c_current_state[0], current_volume, current_time, initial_volume, initial_time)

			#Break the loop cell dead or divided
			if cell_dead >= 0 and cell_divided >= 0:
				warnings.warn("Cell Death and Division Occured Simultaneously - Death Takes Precedent")
				cell_divided = -1
				break
			elif cell_dead >= 0:
				break
			elif cell_divided >= 0:
				break
			#Compute Reaction and Event propensities in-place
			self.interface.compute_lineage_propensities(self.c_current_state, self.c_propensity, current_volume, current_time)

			Lambda = cyrandom.array_sum(&self.c_propensity[0], self.num_propensities)
			# Either we are going to move to the next queued time, or we move to the next reaction time.
			
			if Lambda == 0:
				proposed_time = final_time+1
			else:
				proposed_time = current_time + cyrandom.exponential_rv(Lambda)
			if next_queue_time < proposed_time:
				current_time = next_queue_time
				next_queue_time += delta_t
				move_to_queued_time = 1
			else:
				current_time = proposed_time
				move_to_queued_time = 0
			v.set_time(current_time)


			# Update the results array with the state for the time period that we just jumped through.
			while current_index < num_timepoints and timepoints[current_index] <= current_time:
				for species_index in range(self.num_species):
					self.c_results[current_index, species_index] = self.c_current_state[species_index]
				self.c_volume_trace[current_index] = current_volume
				current_index += 1

			# Now update the state accordingly.
			# IF the queue won, then update the volume and continue on or stop if the cell divided.
			if move_to_queued_time == 1:
				# Update the volume every dtyp
				current_volume = self.interface.apply_volume_rules(&self.c_current_state[0], current_volume, current_time, delta_t)
				v.set_volume(current_volume)

			# if an actual reaction happened, do the reaction and maybe update the queue as well.
			else:
				# select a reaction
				reaction_choice = cyrandom.sample_discrete(self.num_propensities, &self.c_propensity[0], Lambda )
				#Propensities are Ordered:
				# Reactions, Divison Events, Volume Events, Death Events

				#Propensity is a reaction
				if reaction_choice < self.num_reactions:
					# Do the reaction's initial stoichiometry.
					for species_index in range(self.num_species):
						self.c_current_state[species_index] += self.c_stoich[species_index, reaction_choice]

				#Propensity is a VolumeEvent
				elif reaction_choice >= self.num_reactions and reaction_choice < self.num_reactions + self.num_volume_events:
					current_volume = self.interface.apply_volume_event(reaction_choice - self.num_reactions, &self.c_current_state[0], current_time, current_volume)
					v.set_volume(current_volume)
				#Propensity is a DivisionEvent.
				elif reaction_choice >= self.num_reactions+self.num_volume_events and reaction_choice < self.num_reactions + self.num_volume_events+self.num_division_events:
					#Cell Divided = DivisionEvent Index + num_division_rules
					cell_divided = reaction_choice - self.num_reactions - self.num_volume_events + self.num_division_rules
					break
				#Propensity is a Death Event
				elif reaction_choice >= self.num_reactions + self.num_volume_events+self.num_division_events:
					#Cell Divided = DeathEvent Index + num_death_rules
					cell_dead = reaction_choice - self.num_reactions + self.num_volume_events+self.num_division_events+self.num_death_rules
					break
				else:
					raise ValueError("More reaction propensities than expected!")

		#print("Out of SingleCell loop")
		if cell_divided>=0 or cell_dead>=0:
			#Push current state to the nearest index
			if current_time < self.c_truncated_timepoints[current_index]:
				for species_index in range(self.num_species):
					self.c_results[current_index,species_index] = self.c_current_state[species_index]
				self.c_volume_trace[current_index] = current_volume
				current_index += 1

			
			timepoints = timepoints[:current_index]
			self.c_volume_trace = self.c_volume_trace[:current_index]
			self.c_results = self.c_results[:current_index,:]

		#vsr (SingleCellSSAResult) contains the simulation results until cell death / division or simualtion termination.
		#cell_divided and cell_dead are returend via vsr so the events/rules/VolumeSplitters can be called by the lineage simualtion loop.
		SCR = SingleCellSSAResult(np.asarray(timepoints), np.asarray(self.c_results), np.asarray(self.c_volume_trace), cell_divided >= 0)
		SCR.set_divided(cell_divided)
		SCR.set_dead(cell_dead)
		SCR.set_volume_object(v.get_volume_object())
		#("Single Cell Simulation Finished")
		return SCR

	def py_SimulateSingleCell(self, np.ndarray timepoints, LineageModel Model = None, LineageCSimInterface interface = None, LineageVolumeCellState v = None):

		if Model == None and interface == None:
			raise ValueError('py_SimulateSingleCell requires either a LineageModel Model or a LineageCSimInterface interface to be passed in as keyword parameters.')
		elif interface == None:
			interface = LineageCSimInterface(Model)
			interface.py_set_initial_time(timepoints[0])
		if v == None:
			v = LineageVolumeCellState(v0 = 1, t0 = 0, state = Model.get_species_array())

		self.py_set_c_truncated_timepoints(timepoints)
		self.intialize_single_cell_interface(interface)

		return self.SimulateSingleCell(v, self.c_truncated_timepoints)



	#Functions to simulate linages of cells
	#returns a truncated version of the memoryview array[x:]
	#starting at the first index x such that array[x] >= value
	cdef double[:] truncate_timepoints_less_than(self, double[:] array, double value):
		cdef unsigned j = 0
		for j in range(array.shape[0]):
			if array[j] >= value:
				return array[j:]
		return None

	#Functions to simulate linages of cells
	#returns a truncated version of the memoryview array[:x]
	#starting at the first index x such that array[x] > value
	cdef double[:] truncate_timepoints_greater_than(self, double[:] array, double value):
		cdef unsigned j = 0
		for j in range(array.shape[0]):
			if array[j] > value:
				break

		return array[:j]

	#Starts the simulation
	#add_to_lineage toggled for lineage versus propogation simulation
	cdef void simulate_cell_list(self, list initial_cell_states, double[:] timepoints, unsigned add_to_lineage, unsigned create_schnitzes):
		#print("simulate_cell_list")
		cdef unsigned i = 0

		for i in range(len(initial_cell_states)):
			self.r = self.SimulateSingleCell(initial_cell_states[i], timepoints)
			self.old_cell_states.append(self.r.get_final_cell_state())

			if create_schnitzes or add_to_lineage:
				self.s = self.r.get_schnitz()
				self.s.set_parent(None)

				if create_schnitzes:
					self.old_schnitzes.append(self.s)

				if add_to_lineage:
					self.lineage.add_schnitz(self.s)
				
				

	#Simulate inside the Simulation Queue, 1 cell at a time
	#add_to_lineage toggled for lineage versus propogation simulation
	cdef void simulate_daughter_cells(self, double[:] timepoints, unsigned add_to_lineage, unsigned create_schnitzes):
		#print("simulate_daughter_cells")
		cdef double final_time = timepoints[len(timepoints)-1]

		self.r = self.SimulateSingleCell(self.d1, timepoints)
		self.d1final = self.r.get_final_cell_state()

		if add_to_lineage or create_schnitzes:
			self.daughter_schnitz1 = self.r.get_schnitz()

		# Add on the new daughter if final time wasn't reached.
		if self.d1final.get_time() < final_time + 1E-9:
			self.old_cell_states.append(self.d1final)

			if create_schnitzes:
				self.old_schnitzes.append(self.daughter_schnitz1)
		else:
			warnings.warn("Daughter cell simulation went over the total time. Simulation has been discarded. Check for model errors.")

		self.r = self.SimulateSingleCell(self.d2, timepoints)
		self.d2final = self.r.get_final_cell_state()
		
		if add_to_lineage or create_schnitzes:
			self.daughter_schnitz2 = self.r.get_schnitz()

		if self.d2final.get_time() < final_time + 1E-9:
			self.old_cell_states.append(self.d2final)

			if create_schnitzes:
				self.old_schnitzes.append(self.daughter_schnitz2)
		else:
			warnings.warn("Daughter cell simulation went over the total time. Simulation has been discarded. Check for model errors.")

		if add_to_lineage or create_schnitzes:
			# Set up daughters and parent appropriately.
			self.daughter_schnitz1.set_parent(self.s)
			self.daughter_schnitz2.set_parent(self.s)
			self.s.set_daughters(self.daughter_schnitz1,self.daughter_schnitz2)

			if add_to_lineage:
				# Add daughters to the lineage
				self.lineage.add_schnitz(self.daughter_schnitz1)
				self.lineage.add_schnitz(self.daughter_schnitz2)


	#Simulates a lineage of cells keeptring track of mother-daughter relations
	cdef Lineage SimulateCellLineage(self, list initial_cell_states, double[:] timepoints):
		cdef unsigned i, j, list_index
		cdef int cell_divided, cell_dead
		cdef double final_time

		#Check instantation of core data structures
		#Note: these are not automatically instantiated here so they can be reused in more complex simulation types
		if self.lineage is None:
			raise RuntimeError("LineageSSASimulator.lineage must be instantiated to a Lineage before calling SimulateCellLineage. py_SimulateCellLineage automatically does this for you but is slower.")
		if self.old_cell_states is None:
			raise RuntimeError("LineageSSASimulator.old_cell_states must be instantiated to a list before calling SimulateCellLineage. py_SimulateCellLineage automatically does this for you but is slower.")
		if self.old_schnitzes is None:
			raise RuntimeError("LineageSSASimulator.old_schnitzes must be instantiated to a list before calling SimulateCellLineage. py_SimulateCellLineage automatically does this for you but is slower.")


		#These will be used to store outputs
		cdef np.ndarray daughter_cells

		#initialize variables
		final_time = timepoints[timepoints.shape[0]-1]
		i = 0
		list_index = 0
		cell_divided = -1
		cell_dead = -1

		# Simulate the first cell until death division or max time
		
		self.simulate_cell_list(initial_cell_states, timepoints, 1, 1) #toggle add_to_lineage = 1 create snitches 1

		while list_index < len(self.old_cell_states):
			self.cs = self.old_cell_states[list_index]
			self.s = self.old_schnitzes[list_index]

			list_index += 1

			#If the cell has already simulated all its time, do nothing
			if self.cs.get_time() >= final_time- 1E-9:
				pass
			#If the cell is dead, do nothing
			elif self.cs.get_dead() >= 0:
				pass
			#If the cell has divided, apply the appropriate division rule
			elif self.cs.get_divided() >= 0:

				#Check if dt is too small for accurate lineage simulation
				if self.cs.get_initial_time() == self.cs.get_time():
					raise ValueError("Cells are dividing too faster for the timepoints passed into SimulateCellLineage. Try decreasing the spacing between timepoints or limiting cell growth.")

				daughter_cells = self.interface.partition(self.cs.get_divided(), self.cs)

				self.d1 = <LineageVolumeCellState>(daughter_cells[0])
				self.d2 = <LineageVolumeCellState>(daughter_cells[1])

				#Create a new timepoint array and simulate the first daughter and queue if it doesn't reach final time.
				#self.c_truncated_timepoints = timepoints[timepoints >= self.cs.get_time()]
				self.c_truncated_timepoints = self.truncate_timepoints_less_than(timepoints, self.cs.get_time())
				self.simulate_daughter_cells(self.c_truncated_timepoints, 1, 1) #toggle add_to_lineage = 1 and add_schnitz = 1

		return self.lineage

	#Python wrapper of the above
	def py_SimulateCellLineage(self, np.ndarray timepoints, initial_cell_states, LineageCSimInterface interface):
		#Instantiate variables
		self.lineage = Lineage()
		self.old_cell_states = []
		self.old_schnitzes = []
		self.set_c_timepoints(timepoints)
		self.intialize_single_cell_interface(interface)
		return self.SimulateCellLineage(initial_cell_states, self.c_timepoints)


	#Simulates an ensemble of cells over some amount of time.
	#Returns the final cell-states of all cells (and their offspring).
	#  dead cells are included based upon the include_dead_cells parameter (1 = included, otherwise = excluded)
	cdef list PropagateCells(self, list initial_cell_states, double[:] timepoints, unsigned include_dead_cells):
		cdef unsigned list_index

		if self.old_cell_states is None:
			raise RuntimeError("LineageSSASimulator.old_cell_states must be instantiated to a list before calling PropagateCells. py_SimulateCellLineage automatically does this for you but is slower.")
		if self.cell_states is None:
			raise RuntimeError("LineageSSASimulator.cell_states must be instantiated to a list before calling PropagateCells. py_SimulateCellLineage automatically does this for you but is slower.")
		

		cdef double final_time = timepoints[timepoints.shape[0]-1]
		list_index = 0

		#Simulate the initial cells
		self.simulate_cell_list(initial_cell_states, timepoints, 0, 0) #Toggle add to lineage 0 create schnitzes 0

		#Enter Simulation Queue
		while list_index < len(self.old_cell_states):
			self.cs = self.old_cell_states[list_index]
			list_index += 1


			#If the cell is dead, do not simulate. Save if include_dead_cells toggled
			if self.cs.get_dead() >= 0 and include_dead_cells == 1:
				self.cell_states.append(self.cs)

			#If the cell has already simulated all its time, do not simulate and save
			elif self.cs.get_time() >= final_time - final_time*1E-10:
				self.cell_states.append(self.cs)

			#If the cell has divided, apply the appropriate division rule
			elif self.cs.get_divided() >= 0:
				daughter_cells = self.interface.partition(self.cs.get_divided(), self.cs)
				
				self.d1 = <LineageVolumeCellState>(daughter_cells[0])
				self.d2 = <LineageVolumeCellState>(daughter_cells[1])

				#Create a new timepoint array and simulate the first daughter and queue if it doesn't reach final time.
				self.c_truncated_timepoints = self.truncate_timepoints_less_than(timepoints, self.cs.get_time())
				self.simulate_daughter_cells(self.c_truncated_timepoints, 0, 0) #toggle add_to_lineage = 0 create_schnitzes = 0

		return self.cell_states
	#Python wrapper of the above
	def py_PropagateCells(self, np.ndarray timepoints, list initial_cell_states, LineageCSimInterface interface, unsigned include_dead_cells):
		self.cell_states = []
		self.old_cell_states = []
		self.set_c_timepoints(timepoints)
		self.intialize_single_cell_interface(interface)
		return self.PropagateCells(initial_cell_states, timepoints, include_dead_cells)


	#Propogates a single cell trajectory, ignoring half the daughters every division.
	cdef list SingleCellLineage(self, LineageVolumeCellState initial_cell, double[:] timepoints):
		cdef double final_time = timepoints[timepoints.shape[0]-1]
		cdef unsigned list_index = 0
		cdef unsigned i

		self.cell_states = []
		self.old_cell_states = []
		
		#These will be used to store outputs
		cdef np.ndarray daughter_cells


		self.r = self.SimulateSingleCell(initial_cell, timepoints)
		self.old_cell_states.append(self.r.get_final_cell_state())
		self.cell_states.append(self.r)

		while list_index < len(self.old_cell_states):
			self.cs = self.old_cell_states[list_index]
			list_index += 1

			#If the cell has already simulated all its time, do nothing
			if self.cs.get_time() >= final_time- 1E-9:
				pass
			#If the cell is dead, return the list of cells
			elif self.cs.get_dead() >= 0:
				return self.cell_states
			#If the cell has divided, apply the appropriate division rule
			elif self.cs.get_divided() >= 0:
				daughter_cells = self.interface.partition(self.cs.get_divided(), self.cs)

				#Create a new timepoint array and simulate a random daughter and queue if it doesn't reach final time.
				i = <unsigned>cyrandom.uniform_rv()>.5
				self.d1 = <LineageVolumeCellState>(daughter_cells[i])
				self.c_truncated_timepoints = self.truncate_timepoints_less_than(timepoints, self.cs.get_time())
				self.r = self.SimulateSingleCell(self.d1, self.c_truncated_timepoints)
				self.cell_states.append(self.r)
				self.d1final = self.r.get_final_cell_state()

				# Add on the new daughter if final time wasn't reached.
				if self.d1final.get_time() < final_time + 1E-9:
					self.old_cell_states.append(self.d1final)
				else:
					warnings.warn("Daughter cell simulation went over the total time. Simulation has been discarded. Check for model errors.")

		return self.cell_states
	#Python wrapper of the above
	def py_SingleCellLineage(self, np.ndarray timepoints, LineageVolumeCellState initial_cell, LineageCSimInterface interface):
		self.set_c_timepoints(timepoints)
		self.intialize_single_cell_interface(interface)
		return self.SingleCellLineage(initial_cell, timepoints)

	
#Auxilary wrapper functions for quick access to Lineage Simulations

#SingleCellLineage simulates the trajectory of a single cell, randomly discarding one of its daughters every division.
def py_SingleCellLineage(timepoints, initial_cell_state = None, LineageModel Model = None, LineageCSimInterface interface = None, LineageSSASimulator simulator = None, return_dataframes = True):
	if Model == None and interface == None:
		raise ValueError('py_PropagateCells requires either a LineageModel Model or a LineageCSimInterface interface to be passed in as keyword parameters.')
	elif interface == None:
		interface = LineageCSimInterface(Model)
		interface.py_set_initial_time(timepoints[0])

	if initial_cell_state is None:
		initial_cell_state = LineageVolumeCellState(v0 = 1, t0 = 0, state = interface.py_get_initial_state())
	elif not isinstance(initial_cell_state, LineageVolumeCellState):
		raise ValueError("initial_cell_state must be of type LineageVolumeCellState or None (in which case it will default to the Model's initial state).")

	if simulator == None:
		simulator = LineageSSASimulator()

	result = simulator.py_SingleCellLineage(timepoints, initial_cell_state, interface)

	cell_lineage = []
	if return_dataframes:#Converts list of cell states into a Pandas dataframe
		try:
			import pandas
			df_list = [r.py_get_dataframe(Model = Model) for r in result]
			return pandas.concat(df_list)
		except ModuleNotFoundError:
			warnings.warn("return_dataframes=True requires that pandas be installed. Instead a numpy array is being returned (each column is a species, the last column is volume, and rows are cell states)")
	else:
		return result

#py_PropagateCells simulates an ensemble of growing dividing cells, returning only the cell states at the end of timepoints
#include_dead_cells toggles whether all dead cells accumulated along the way will also be returned.
#return data_frames returns all the results as a pandas dataframe. Otherwise results are returned as a list of LineageVolumeCellStates
def  py_PropagateCells(timepoints, initial_cell_states = [], LineageModel Model = None, LineageCSimInterface interface = None, LineageSSASimulator simulator = None, include_dead_cells = False, return_dataframes = True):
	
	if Model == None and interface == None:
		raise ValueError('py_PropagateCells requires either a LineageModel Model or a LineageCSimInterface interface to be passed in as keyword parameters.')
	elif interface == None:
		interface = LineageCSimInterface(Model)
		interface.py_set_initial_time(timepoints[0])

	if isinstance(initial_cell_states, int):
		initial_cell_states = [LineageVolumeCellState(v0 = 1, t0 = 0, state = interface.py_get_initial_state())]*initial_cell_states
	elif (isinstance(initial_cell_states, list) and len(initial_cell_states) == 0):
		initial_cell_states = [LineageVolumeCellState(v0 = 1, t0 = 0, state = interface.py_get_initial_state())]
	elif not isinstance(initial_cell_states, list):
		raise ValueError("Initial Cell States must be a list of LineageVolumeCell states or and positive integer")
	if simulator == None:
		simulator = LineageSSASimulator()

	final_cell_states = simulator.py_PropagateCells(timepoints, initial_cell_states, interface, include_dead_cells)

	if return_dataframes:#Converts list of cell states into a Pandas dataframe
		try:
			import pandas
			darray = np.array([np.append(cs.py_get_state(), cs.py_get_volume()) for cs in final_cell_states])
			if Model == None:
				warnings.warn("Without passing in a model, the data frame will not be indexable by species name.")
				df = pandas.DataFrame(darray)
			else:
				columns = Model.get_species_list()+["volume"]
				df = pandas.DataFrame(darray, columns = columns)
			return df
		except ModuleNotFoundError:
			warnings.warn("return_dataframes=True requires that pandas be installed. Instead a numpy array is being returned (each column is a species, the last column is volume, and rows are cell states)")
	else:
		return final_cell_states

#SimulateCellLineage simulates a lineage of growing, dividing, and dieing cells over timepoints. 
#The entire time trajectory of the simulation is returned as a Lineage which contains a binary tree of Schnitzes each containing a LineageSingleCellSSAResult.
def py_SimulateCellLineage(timepoints, initial_cell_states = [], initial_cell_count = 1, interface = None, Model = None):

	simulator = LineageSSASimulator()
	if Model == None and interface == None:
		raise ValueError('py_SimulateCellLineage requires either a LineageModel Model or a LineageCSimInterface interface to be passed in as keyword parameters.')
	elif interface == None:
		interface = LineageCSimInterface(Model)
		interface.py_set_initial_time(timepoints[0])

	if isinstance(initial_cell_states, int):
		initial_cell_states = [LineageVolumeCellState(v0 = 1, t0 = 0, state = interface.py_get_initial_state())]*initial_cell_states
	elif (isinstance(initial_cell_states, list) and len(initial_cell_states) == 0):
		initial_cell_states = [LineageVolumeCellState(v0 = 1, t0 = 0, state = interface.py_get_initial_state())]
	elif not isinstance(initial_cell_states, list):
		raise ValueError("Initial Cell States must be a list of LineageVolumeCell states or and positive integer")

	return simulator.py_SimulateCellLineage(timepoints, interface = interface, initial_cell_states = initial_cell_states)


#SimulateSingleCell performs an SSA simulation on a single cell until it divides, dies, or the final timepoint arrives.
def py_SimulateSingleCell(timepoints, Model = None, interface = None, initial_cell_state = None, return_dataframes = True):
	simulator = LineageSSASimulator()
	if Model == None and interface == None:
		raise ValueError('py_SimulateSingleCell requires either a LineageModel Model or a LineageCSimInterface interface to be passed in as keyword parameters.')
	elif interface == None:
		interface = LineageCSimInterface(Model)
		interface.py_set_initial_time(timepoints[0])

	if initial_cell_state == None:
		v = LineageVolumeCellState(v0 = 1, t0 = 0, state = interface.py_get_initial_state())
	
	result = simulator.py_SimulateSingleCell(timepoints, Model = Model, interface = interface, v = v)

	if return_dataframes:
		return result.py_get_dataframe(Model = Model)
	else:
		return result


#A simulator class for interacting cell lineages
cdef class InteractingLineageSSASimulator(LineageSSASimulator):

	#Used for Simulating Interacting lineages
	cdef int spec_ind,
	cdef unsigned num_global_species, num_interfaces, total_cell_count
	cdef double[:] global_species, c_period_timepoints
	cdef int[:, :] global_species_inds #stores global_species_inds[i, j] --> species index of interface j for global species i

	cdef double total_cell_volume, global_volume, leftover_global_volume, temp_volume, global_volume_param, average_dist_threshold, global_sync_period, dt

	cdef list interface_list, new_schnitzes, new_cell_states, lineage_list, old_cell_state_list, new_cell_state_list, old_schnitz_list, new_schnitz_list
	cdef SingleCellSSAResult new_r, merge_r
	cdef LineageVolumeCellState new_cs
	cdef Schnitz new_s


	#Initializes a new interface and sets all internal variables such as lineage and cell_lists appropriately
	cdef void switch_interface(self, unsigned interface_ind):
		#print("switch_interface", interface_ind)
		self.intialize_single_cell_interface(self.interface_list[interface_ind])
		self.lineage = self.lineage_list[interface_ind]
		self.old_cell_states = self.old_cell_state_list[interface_ind]
		self.new_cell_states = self.new_cell_state_list[interface_ind]
		self.old_schnitzes = self.old_schnitz_list[interface_ind]
		self.new_schnitzes = self.new_schnitz_list[interface_ind]

	#Calculates volume variables from the entire population of cells
	cdef void calculate_global_volumes(self):
		#print("calculate_global_volumes")
		cdef unsigned interface_ind = 0
		cdef unsigned i = 0
		self.total_cell_count = 0
		self.total_cell_volume = 0 #reset total cell volume

		for interface_ind in range(self.num_interfaces):

			self.old_cell_states = self.old_cell_state_list[interface_ind]

			for i in range(len(self.old_cell_states)):
				self.cs = self.old_cell_states[i]
				self.total_cell_volume += self.cs.get_volume()
				self.total_cell_count += 1

		if self.global_volume_param == 0: #If global volume is 0, assume global_volume = total_cell_volume for the entire simulation
			self.leftover_global_volume = 0
			self.global_volume = self.total_cell_volume
		elif self.total_cell_volume > self.global_volume_param:
			warnings.warn("Total cell volume exceeded global volume. All cells set to dead and simulation terminated.")
			#Set all cells to dead
			#Death by crowding
			for interface_ind in range(self.num_interfaces):
				self.old_cell_states = self.old_cell_state_list[interface_ind]
				for i in range(len(self.old_cell_states)):
					self.cs = self.old_cell_states[i]
					self.cs.set_dead(1)
		else:
			self.global_volume = self.global_volume_param
			self.leftover_global_volume = self.global_volume - self.total_cell_volume

	
	#Calculates the total number of global species across cell states
	cdef void calculate_global_species_totals(self):
		#print("calculate_global_species_totals")
		cdef unsigned interface_ind = 0
		cdef unsigned list_index = 0
		cdef unsigned i = 0
		cdef int spec_ind = 0

		#Cycle through all the cell_states by interface
		for interface_ind in range(self.num_interfaces):
			self.old_cell_states = self.old_cell_state_list[interface_ind]
			#cycle through each cell
			for list_index in range(len(self.old_cell_states)):
				self.cs = self.old_cell_states[list_index]
				self.c_current_state = self.cs.get_state()
				#Add the global species if they are in the model
				for i in range(self.num_global_species):
					spec_ind = self.global_species_inds[i, interface_ind]
					if spec_ind >= 0: #If the cell contains the global species, set its internal s_i to 0 and add that to the global total
						self.global_species[i] += self.c_current_state[spec_ind]
						self.cs.set_state_comp(0, spec_ind)

	#Synchronizes global species by redistributing them between different volumes, including the global volume
	cdef void synchronize_global_species(self):
		#print("synchronize_global_species")
		cdef unsigned i = 0

		#Calculate the global volumes
		self.calculate_global_volumes()
		#Calculate the global species totals
		self.calculate_global_species_totals()

		for i in range(self.num_global_species):
			#If the amount of a global species is above the threshold for stochastic distribute, distribute the average
			if self.total_cell_volume/self.global_volume*self.global_species[i]/self.total_cell_count > self.average_dist_threshold:
				self.global_species[i] = self.distribute_global_species_average(self.global_species[i])

			#Otherwise distribute stochastically
			else:
				self.global_species[i] = self.distribute_global_species_multinomial(self.global_species[i])

	#Distribute global species to their expected values	
	#global_species is the number of species to distribue between all old_cell_states
	#will distribute these species and return the number of species left in the global volume
	cdef double distribute_global_species_average(self, double global_count):
		#print("distribute_global_species_average")
		cdef unsigned i = 0
		cdef unsigned temp_count = 0
		cdef double new_global_species = global_count  #stores the number of global species passed to the global volume
		cdef unsigned list_index = 0
		cdef unsigned interface_ind = 0
		cdef int spec_ind = 0
				
		for interface_ind in range(self.num_interfaces):
			self.old_cell_states = self.old_cell_state_list[interface_ind]
			for list_index in range(len(self.old_cell_states)):
				self.cs = self.old_cell_states[list_index]
				temp_count = int(self.cs.get_volume()/self.global_volume*global_count)

				spec_ind = self.global_species_inds[i, interface_ind]
				#Add the global species if they are in the model
				if spec_ind >= 0: #Check if the cell contains that species
					new_global_species -= temp_count
					self.cs.set_state_comp(temp_count, spec_ind)

		return new_global_species
	
	#Distribute global species stochastically	
	#global_count is the number of species to distribue between all old_cell_states
	#will distribute these species and return the number of species left in the global volume
	cdef double distribute_global_species_multinomial(self, double global_count):
		#print("distribute_global_species_multinomial")
		cdef unsigned i = 0
		cdef double rand
		cdef double temp_volume = 0
		cdef double new_global_species = 0  #stores the number of global species passed to the global volume
		cdef unsigned list_index = 0
		cdef unsigned interface_ind = 0
		cdef int spec_ind = 0

		while global_count > 0:
			rand = cyrandom.uniform_rv()
			temp_volume = self.leftover_global_volume
			#randomly add to the global volume first because it is more probable
			if rand <= temp_volume/self.global_volume:
				new_global_species += 1
				global_count -= 1
				continue

			#cycle through cells
			for interface_ind in range(self.num_interfaces):
				self.old_cell_states = self.old_cell_state_list[interface_ind]
				for list_index in range(len(self.old_cell_states)):
					self.cs = self.old_cell_states[list_index]
					temp_volume += self.cs.get_volume()
					if rand <= temp_volume/self.global_volume:
						spec_ind = self.global_species_inds[i, interface_ind]
						if spec_ind >= 0: #Check if the cell contains that species
							self.c_current_state = self.cs.get_state()
							self.cs.set_state_comp(self.c_current_state[spec_ind]+1, spec_ind) #add one to the cell state
							global_count -= 1 #decrement the global species
						else: #if the cell doesn't contain the species, add it to the new global species vector
							new_global_species += 1
							global_count -= 1
						break

		return new_global_species
	
	#Helper function to simulate one sync-period of an interacting lineage
	cdef void SimulateInteractingLineagePeriod(self, double[:] timepoints):
		#print("SimulateInteractingLineagePeriod")
		cdef np.ndarray daughter_cells
		cdef unsigned interface_ind = 0
		cdef unsigned list_index = 0
		cdef double final_time = timepoints[timepoints.shape[0]-1]
		#Cycle through interfaces
		for interface_ind in range(self.num_interfaces):
			self.switch_interface(interface_ind)#set the correct interface and internal variables

			#cycle through cells
			#print("entering while loop")
			while list_index < len(self.old_cell_states): 
				self.cs = self.old_cell_states[list_index]
				self.s = self.old_schnitzes[list_index]
				list_index += 1

				#If the cell is dead add it to the lineage
				if self.cs.get_dead() >= 0:
					# Add daughters to the lineage
					self.lineage.add_schnitz(self.s)

				#If the cell has divided right at teh end of the period, add it to the next period for division
				#Do not add to lineage because that will happen next period
				elif self.cs.get_divided() >= 0 and self.cs.get_time() >= final_time - 1E-9:
					self.new_cell_states.append(self.cs)
					self.new_schnitzes.append(self.s)

				#If a cell has divided and still has time left in the period, simulate the daughters.
				elif self.cs.get_divided() >= 0:
					self.lineage.add_schnitz(self.s)
					daughter_cells = self.interface.partition(self.cs.get_divided(), self.cs)
					self.d1 = <LineageVolumeCellState>(daughter_cells[0])
					self.d2 = <LineageVolumeCellState>(daughter_cells[1])
					self.c_truncated_timepoints = self.truncate_timepoints_less_than(timepoints, self.cs.get_time())
					self.simulate_daughter_cells(self.c_truncated_timepoints, 0, 1) #Toggle add to lineage False and create schnitzes True

				#If the cell has reached its period time, add it to new_schnitzes and cell_states
				elif self.cs.get_time() >= final_time - 1E-9:
					self.new_schnitzes.append(self.s)
					self.new_cell_states.append(self.cs)

				#If the cell isn't dead or divided or at period time simulate it more
				else:
					#If there is only one timepoint left, push to the next period.
					self.c_truncated_timepoints = self.truncate_timepoints_less_than(timepoints, self.cs.get_time())
				
					if len(self.c_truncated_timepoints) <= 2:
						#print("only one timepoint left, push by deltaT")
						self.new_schnitzes.append(self.s)
						self.new_cell_states.append(self.cs)

					else:
						self.new_r = self.SimulateSingleCell(self.cs, self.c_truncated_timepoints)
						self.new_cs = self.new_r.get_final_cell_state()
						self.new_cs.set_initial_vars(self.cs.get_initial_volume(), self.cs.get_initial_time())
						self.new_cs.set_time(self.cs.get_time())
						self.new_cs.set_volume(self.cs.get_volume())

						#After simulation, merge the two SSA results
						self.merge_r = SingleCellSSAResult(np.concatenate((self.s.get_time(), self.new_r.get_timepoints())), 
							np.concatenate((self.s.get_data(), self.new_r.get_result())), 
							np.concatenate((self.s.get_volume(), self.new_r.get_volume())), 
							self.new_r.get_divided() >= 0)

						

						self.merge_r.set_divided(self.new_r.get_divided())
						self.merge_r.set_dead(self.new_r.get_dead())
						self.new_s = self.merge_r.get_schnitz()
						self.new_s.set_parent(self.s.get_parent())

						
						#Add schnitzes to the next period if they are done simulating
						if self.new_cs.get_time() >= final_time - 1E-9:
							self.new_schnitzes.append(self.new_s)
							self.new_cell_states.append(self.new_cs)
						#stay in the same period (perhaps they have died or divided)
						else:
							self.old_schnitzes.append(self.new_s)
							self.old_cell_states.append(self.new_cs)


			#After going through all the old_cell_states, make new_cell_states old.

			self.old_cell_state_list[interface_ind] = self.new_cell_states
			self.old_schnitz_list[interface_ind] = self.new_schnitzes
			self.new_cell_state_list[interface_ind] = []
			self.new_schnitz_list[interface_ind] = []
			#print("End of Period Loope")

			

	cdef list SimulateInteractingCellLineage(self, list interface_list, list initial_cell_states, double[:] timepoints, double global_sync_period, np.ndarray global_species_inds, double global_volume_param, double average_dist_threshold):
		#print("Starting Interacting Lineage Simulation")

		cdef unsigned i = 0
		cdef unsigned j = 0
		cdef unsigned spec_ind = 0
		cdef unsigned list_index = 0
		cdef unsigned interface_ind = 0
		
		cdef double final_time = timepoints[timepoints.shape[0]-1] #when the entire simulation ends (adding dt onto the end for rounding reasons)
		cdef double current_time = timepoints[0] #current time
		
		self.dt = timepoints[1] - timepoints[0]
		self.global_sync_period = global_sync_period #How often global species are synchronized
		cdef double period_time = timepoints[0]+self.global_sync_period #When the next sync period happens

		#Store seperate cell lists and lineages for each interface
		self.lineage_list = [] #stores one lineage for each interface
		self.old_cell_state_list = [] #stores one list of cell states for each interface
		self.new_cell_state_list = [] #stores one list of new cell states for each interface
		self.old_schnitz_list = [] #stores one list of schnitzes for each interface
		self.new_schnitz_list = []

		self.global_species_inds = global_species_inds #stores global_species_inds[i, j] --> species index of interface j for global species i

		#These parameters are global because they are used by helper functions
		self.total_cell_volume = 0 #stores sum_i volume(cell_i)
		self.num_global_species = global_species_inds.shape[0]
		self.global_species = np.zeros(self.num_global_species) #stores the global species vector
		self.leftover_global_volume = 0 #stores global_volume - total_cell_volume
		self.global_volume = 0 #stores global_volume_param OR total_cell_volume if global_volume_param == 0.
		self.global_volume_param = global_volume_param
		self.average_dist_threshold = average_dist_threshold

		

		#Check that len sims == len initial_cells. As cells divide, they inherit their mother's CSimInterface
		#Now done in python wrappers
		#if len(self.interface_list) != len(initial_cell_states):
		#	raise ValueError(f"interface list (length {len(self.interface_list)}) [a list of LineageCSimInterfaces] must be the same length as initial cells (length {len(initial_cell_states)}) [[a list of LineageVolumeCellStates] for each interface].")
		#Check that global period is greater than dt
		if self.global_sync_period < self.dt:
			raise ValueError("global sync period must be larger than the timestep, dt, in the timepoints passed in for simulation.")
		#Check global sync period is smaller than the total time simulated
		if self.global_sync_period >= final_time:
			raise ValueError("global sync period must be smaller than the entire length of the timepoints passed in for simulation.")
		
		#Calculate global volume
		self.calculate_global_volumes()

		self.interface_list = interface_list
		self.num_interfaces = len(self.interface_list)
		for interface_ind in range(self.num_interfaces):
			self.lineage_list.append(Lineage())
			self.new_cell_state_list.append([])
			self.old_cell_state_list.append([])
			self.old_schnitz_list.append([])
			self.new_schnitz_list.append([])

		#Timepoints for the first set of simulations
		self.c_period_timepoints = self.truncate_timepoints_greater_than(timepoints, period_time)

		#Simulate the initial cell states
		#All cells simulated till either they die, divide or they hit period time
		#Cell states placed in self.old_cell_state_list[interface_ind] organized by interface_ind
		for interface_ind in range(self.num_interfaces):
			self.switch_interface(interface_ind) #Very important to call this before the simulation
			self.simulate_cell_list(initial_cell_states[interface_ind], self.c_period_timepoints, 0, 1) #toggle add_to_lineage = 0 create_schnitzes = 1

		#Main Simulation loop
		#print("Entering Main loop")
		while period_time <= final_time and current_time <= final_time:
			#Update timepoints
			current_time = period_time

		
			#Do nothing and the loop will end
			if period_time > final_time:
				pass
			#If the total sim time isn't a multiple of the period time, update carefully
			elif period_time + self.global_sync_period >= final_time and period_time < final_time:
				period_time = final_time+self.dt
			#Otherwise update normally
			elif period_time + self.global_sync_period < final_time+self.dt:
				period_time = period_time + self.global_sync_period

			self.c_period_timepoints = self.truncate_timepoints_greater_than(timepoints, period_time)

			#Calculate global volume and synchronize global species across all cells in self.old_cell_state_list
			self.synchronize_global_species() #calculates global values and redistributes the species
			self.SimulateInteractingLineagePeriod(self.c_period_timepoints)

			self.new_schnitz_list = []
			self.new_cell_state_list = []
			for interface_ind in range(self.num_interfaces):
				self.new_schnitz_list.append([])
				self.new_cell_state_list.append([])

		#Add the final schnitzes to their lineages
		for interface_ind in range(self.num_interfaces):
			self.switch_interface(interface_ind)
			for list_index in range(len(self.old_schnitzes)): 
				self.s = self.old_schnitzes[list_index]
				self.lineage.add_schnitz(self.s)

		return self.lineage_list

	#Python accessor
	def py_SimulateInteractingCellLineage(self, np.ndarray timepoints, list interface_list, list initial_cell_states, double global_sync_period, np.ndarray global_species_inds, double global_volume_param, double average_dist_threshold):
		self.set_c_timepoints(timepoints)
		#print("py_SimulateInteractingCellLineage 2: timepoints.shape",timepoints.shape, timepoints[0], timepoints[timepoints.shape[0]-1]) 
		return self.SimulateInteractingCellLineage(interface_list, initial_cell_states, timepoints, global_sync_period, global_species_inds, global_volume_param, average_dist_threshold)


#Auxilary Python Function
def py_SimulateInteractingCellLineage(timepoints, global_sync_period, global_species = [], interface_list = [], model_list = [], initial_cell_states = [], simulator = None, global_species_inds = None, global_volume = 0, average_dist_threshold = 1.0):
	if simulator == None:
		simulator = InteractingLineageSSASimulator()
	
	if len(model_list) == 0 and len(interface_list) == 0:
		raise ValueError("Missing Required Keyword Arguments:models = [LineageModel] or interface_list = [LineageCSimInterface]")
	elif len(interface_list) == 0:
		interface_list = [LineageCSimInterface(m) for m in model_list]
		global_species_inds = np.zeros((len(global_species), len(model_list)))
		for i in range(len(global_species)):
			for j in range(len(model_list)):
				m = model_list[j]
				s = global_species[i]
				ind = m.get_species_index(s)
				if ind != None:
					global_species_inds[i, j] = ind
				else:
					global_species_inds[i, j] = -1
	elif len(model_list) != 0 and len(interface_list) != 0:
		raise ValueError("Must call py_SimulateInteractingCellLineage with either the keyword argument model_list or the keyword argument interface_list, not both.")
	elif len(model_list) == 0 and (global_species_inds == None or global_species_inds.shape[1] != len(interface_list)):
		raise ValueError("When calling py_SimulateInteractingCellLineage with the keyword argument interface_list, the argument global_species_inds is required where global_species_inds[i, j] corresponds the species index of the ith global species in the jth interface.")
	elif len(global_species) == 0 and global_species_inds == None:
		warnings.warn('Calling SimulateInteractintCellLineage without any global species defined. Use the global_species or global_species_inds keywords.')
		global_species_inds = np.array()


	if len(initial_cell_states) == len(interface_list) and  isinstance(initial_cell_states[0], int):
		initial_cell_counts = initial_cell_states
		initial_cell_states = []

		for i in range(len(interface_list)):
			initial_cell_states.append([])
			interface = interface_list[i]
			initial_state = interface.py_get_initial_state()
			for j in range(initial_cell_counts[i]):
				lvcs = LineageVolumeCellState(v0 = 1.0, t0 = timepoints[0], state = initial_state.copy())
				initial_cell_states[i].append(lvcs)

	elif len(initial_cell_states) == 0 and len(interface_list) > 0:
		warnings.warn("Calling py_SimulateInteractintCellLineage without any initial_cell_states. Defaulting to creating one initial cell for each interface.")
		initial_cell_states = [LineageVolumeCellState(v0 = 1, t0 = 0, state = i.py_get_initial_state()) for i in interface_list]

	elif len(initial_cell_states) > len(interface_list):
		raise ValueError("When passing in more initial_cell_states than models, the keyword argument interface_inds is also required where interface_inds[i] corresponds to the index of the interface/model beloning to initial_cell_state[i].")
	
	return simulator.py_SimulateInteractingCellLineage(timepoints, interface_list, initial_cell_states, global_sync_period, global_species_inds.astype(int), global_volume, average_dist_threshold)



		

		






#Inputs:
#  initial_cells: a list of LineageVolumeCellStates
#  sims: a list of LineageCsimInterfaces.
#  interface_inds is a mapping of from the initial_cell index --> LineageCSimInterface index [in the sims list]
#  global_sync_period: global species are synchronized between the cell ensemble every global_sync_period
#  global_species_indices: an (# global species x # interfaces) array of species indices for each model (interface)
#  global_volume_param: the volume of the global container. If 0, global_volume = total_cell_volume. Otherwise simulation ends when total_cell_volume > global_volume
#  average_dist_threshold: The expected fraction of molecules per cell to switch between perfect geometric distribution and average distribution. 
#       High values >> 1 can dramatically slow down the simulation. Low values < 1 might result in no distribution of global species.
# Returns a Lineage


"""
cdef Lineage SimulateInteractingCellLineage(list sim_interfaces, list initial_cell_states, list interface_inds, np.ndarray timepoints, LineageSSASimulator simulator, 
	double global_sync_period, np.ndarray global_species_inds, double global_volume_param, double average_dist_threshold):
	
	


	# Simulate the initial cells until death division or synch-time


	total_cell_volume = 0 #reset total cell volume after simulation
	



	#Global Simulation Loop till final_time
	#print("Entering Queue Loop")
	
		#print("final global_speices = ", global_species)
		#enter simulation queue
		list_index = 0
		while list_index < len(old_cell_states):
			#print("Simulating list index:", list_index, "len(old_cell_states)", len(old_cell_states))
			cs = old_cell_states[list_index]
			s = old_schnitzes[list_index]
			i = old_interface_inds[list_index]
			sim = sim_interfaces[i]
			list_index += 1

			total_cell_volume = 0 #reset total cell volume after simulation

			#If the cell is dead or it has simulated to the final time, add it to the lineage
			if cs.get_dead() >= 0 or cs.get_time() >= final_time - 1E-9:
				#print("simulation complete")
				# Add daughters to the lineage
				l.add_schnitz(s)
			#If the cell has divided, apply the appropriate division rule
			elif cs.get_divided() >= 0:
				#print('cell divided - adding daughters')

				c_truncated_timepoints = c_period_timepoints[c_period_timepoints > cs.get_time()]

				#sometimes a cell divides at the very end of a period - push it to the next period's queue.
				if len(c_truncated_timepoints) <= 1:
					new_schnitzes.append(s)
					new_interface_inds.append(i)
					new_cell_states.append(cs)

				#Otherwise, divide the cell
				else:
					#print("cell already divided, add to lineage")
					#Add mother to the lineage
					l.add_schnitz(s)

					daughter_cells = sim.partition(cs.get_divided(), cs)
					
					d1 = <LineageVolumeCellState>(daughter_cells[0])
					d2 = <LineageVolumeCellState>(daughter_cells[1])

					#Create a new timepoint array and simulate the first daughter
					
					#print("c_truncated_timepoints", c_truncated_timepoints[0], c_truncated_timepoints[len(c_truncated_timepoints)-1])
					r = simulator.SimulateSingleCell(sim, d1, c_truncated_timepoints)
					daughter_schnitz1 = r.get_schnitz()
					d1final = r.get_final_cell_state()

					#simulate the second daughter
					r = simulator.SimulateSingleCell(sim, d2, c_truncated_timepoints)
					daughter_schnitz2 = r.get_schnitz()
					d2final = r.get_final_cell_state()

					# Set up daughters and parent appropriately.
					daughter_schnitz1.set_parent(s)
					daughter_schnitz2.set_parent(s)
					s.set_daughters(daughter_schnitz1,daughter_schnitz2)

					# Add the new daughter1 cell to the next periods queue if the final period time hasn't been reached
					if d1final.get_time() < c_timepoints[len(c_period_timepoints)] and d1final.get_time() >= c_timepoints[len(c_period_timepoints)-1]:
						new_schnitzes.append(daughter_schnitz1)
						new_cell_states.append(d1final)
						new_interface_inds.append(i)
					# Otherwise continue in the same queue
					else:
						old_schnitzes.append(daughter_schnitz1)
						old_cell_states.append(d1final)
						old_interface_inds.append(i)

					# Add the new daughter2 cell to the next periods queue if the final period time hasn't been reached
					if d2final.get_time() < c_timepoints[len(c_period_timepoints)] and d2final.get_time() >= c_timepoints[len(c_period_timepoints)-1]:
						new_schnitzes.append(daughter_schnitz2)
						new_cell_states.append(d2final)
						new_interface_inds.append(i)

					# Otherwise continue in the same queue
					else:
						old_schnitzes.append(daughter_schnitz2)
						old_cell_states.append(d2final)
						old_interface_inds.append(i)

			#If the cell isn't dead or divided, simulate it more
			else:
				#If there is only one timepoint left, push to the next period.
				c_truncated_timepoints = c_period_timepoints[c_period_timepoints > cs.get_time()]
				#print("period_Time", period_time, "len(c_period_timepoints)", len(c_period_timepoints), "c_period_timepoints[0]=", c_period_timepoints[0], "c_period_timepoints[-1]=", c_period_timepoints[len(c_period_timepoints)-1])
				#print("cs.get_time()=", cs.get_time(), "len(c_truncated_timepoints)", len(c_truncated_timepoints), "c_truncated_timepoints[0]=", c_truncated_timepoints[0], "c_truncated_timepoints[-1]=", c_truncated_timepoints[len(c_truncated_timepoints)-1])


				if len(c_truncated_timepoints) <= 1:
					#print("only one timepoint left, push by deltaT")
					new_schnitzes.append(s)
					new_cell_states.append(cs)
					new_interface_inds.append(i)

				else:
					#print("continuation simulation")
					new_r = simulator.SimulateSingleCell(sim, cs, c_truncated_timepoints)
					#print("SSA Complete")
					new_cs = new_r.get_final_cell_state()
					new_cs.set_initial_vars(cs.get_initial_volume(), cs.get_initial_time())
					new_cs.set_time(cs.get_time())
					new_cs.set_volume(cs.get_volume())

					#print("merging schnitzes: s.get_time()[0]=", s.get_time()[0], "s.get_time()[-1]=",s.get_time()[len(s.get_time())-1], " new_r.get_timepoints()[0]",  new_r.get_timepoints()[0], " new_r.get_timepoints()[-1]",  new_r.get_timepoints()[len( new_r.get_timepoints())-1])
					merge_r = SingleCellSSAResult(
						np.concatenate((s.get_time(), new_r.get_timepoints())), 
						np.concatenate((s.get_data(), new_r.get_result())), 
						np.concatenate((s.get_volume(), new_r.get_volume())), 
						new_r.get_divided() >= 0)

					merge_r.set_divided(new_r.get_divided())
					merge_r.set_dead(new_r.get_dead())
					new_s = merge_r.get_schnitz()
					new_s.set_parent(s.get_parent())
					
					#print("ssa results merged")

					#print("adding to lists")
					#Save final Schnitz
					if new_cs.get_time() >= final_time:
						#print("Saving Final Schnitz", "new_cs.get_time()", new_cs.get_time())
						l.add_schnitz(new_s)
					#move to the next time period
					elif new_cs.get_time() <= period_time-1E-8 and new_cs.get_time() >= c_truncated_timepoints[len(c_truncated_timepoints)-1]:
						#print("Add Schnitz to next Period", new_cs.get_time(), period_time)
						new_schnitzes.append(new_s)
						new_cell_states.append(new_cs)
						new_interface_inds.append(i)
					#stay in the same period
					else:
						#print('Adding Schnitz to to current queue', new_cs.get_time(), period_time)
						old_schnitzes.append(new_s)
						old_cell_states.append(new_cs)
						old_interface_inds.append(i)
						
					#print("continuation sim complete")
			#print("end of while loop period_time=", period_time, "final_time=", final_time)
		#print("while loop complete: period_time=", period_time, "final_time=", final_time)
		#reset lists
		old_cell_states = new_cell_states
		new_cell_states = []
		old_schnitzes = new_schnitzes
		new_schnitzes = []
		old_interface_inds = new_interface_inds
		new_interface_inds = []


	return l


def py_SimulateInteractingCellLineage(timepoints, global_sync_period, global_species = [], sim_interfaces = [], models = [], initial_cell_states = [], interface_inds = [], simulator = None, 
	global_species_inds = None, global_volume = 0, average_dist_threshold = 1.0):
	if simulator == None:
		simulator = LineageSSASimulator()
	if len(models) == 0 and len(sim_interfaces) == 0:
		raise ValueError("Missing Required Keyword Arguments:models = [LineageModel] or sim_interfaces = [LineageCSimInterface]")
	elif len(sim_interfaces) == 0:
		sim_interfaces = [LineageCSimInterface(m) for m in models]
		global_species_inds = np.zeros((len(global_species), len(models)))
		for i in range(len(global_species)):
			for j in range(len(models)):
				m = models[j]
				s = global_species[i]
				ind = m.get_species_index(s)
				if ind != None:
					global_species_inds[i, j] = ind
				else:
					global_species_inds[i, j] = -1
	elif len(models) != 0 and len(sim_interfaces) != 0:
		raise ValueError("Must call py_SimulateInteractingCellLineage with either the keyword argument models or the keyword argument sim_interfaces, not both.")
	elif len(models) == 0 and (global_species_inds == None or global_species_inds.shape[1] != len(sim_interfaces)):
		raise ValueError("When calling py_SimulateInteractingCellLineage with the keyword argument sim_interfaces, the argument global_species_inds is required where global_species_inds[i, j] corresponds the species index of the ith global species in the jth interface.")
	elif len(global_species) == 0 and global_species_inds == None:
		warnings.warn('Calling SimulateInteractintCellLineage without any global species defined. Use the global_species or global_species_inds keywords.')
		global_species_inds = np.array()


	if len(initial_cell_states) == len(models) and (not isinstance(initial_cell_states[0], VolumeCellState)):
		initial_cell_counts = initial_cell_states
		initial_cell_states = []
		interface_inds = []
		for i in range(len(models)):
			M = models[i]
			initial_state = M.get_species_array()
			for j in range(initial_cell_counts[i]):
				lvcs = LineageVolumeCellState(v0 = 1, t0 = 0, state = initial_state)
				initial_cell_states.append(lvcs)
				interface_inds.append(i)
	elif len(initial_cell_states) == 0 and len(models) > 0:
		warnings.warn("Calling py_SimulateInteractintCellLineage without any initial_cell_states. Defaulting to creating one initial cell for each model.")
		initial_cell_states = [LineageVolumeCellState(v0 = 1, t0 = 0, state = m.get_species_array()) for m in models]
		interface_inds = [i for i in range(len(models))]
	elif len(initial_cell_states) > len(models) and len(interface_inds) != len(initial_cell_states):
		raise ValueError("When passing in more initial_cell_states than models, the keyword argument interface_inds is also required where interface_inds[i] corresponds to the index of the interface/model beloning to initial_cell_state[i].")
	
	return SimulateInteractingCellLineage(sim_interfaces, initial_cell_states, interface_inds, timepoints,simulator, global_sync_period, global_species_inds, global_volume, average_dist_threshold)

"""