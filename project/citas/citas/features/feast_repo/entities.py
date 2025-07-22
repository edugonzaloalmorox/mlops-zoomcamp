from feast import Entity, ValueType

Codigo_Cita = Entity(name="Codigo_Cita",
                     join_keys=["Codigo_Cita"], 
                     value_type=ValueType.STRING)