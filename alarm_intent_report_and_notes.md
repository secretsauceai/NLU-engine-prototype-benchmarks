intent: alarm_remove, total count: 120, total incorrect count: 41
 example of correctly predicted utterance: cancel my [time : seven am] alarm
 example of incorrectly predicted utterance: stop [time : seven am] alarm

 NOTES: would a user say stop an alarm to remove it? I think stopping an alarm would be one already going off, which isn't remove.

 I think we should remove any remove_alarm that isn't strictly about removing an alarm already set. Also it seems like the predicted intents are often wrong for a lot of them. Perhaps refining the data will help.

incorrect predicted intents for alarm_remove and their counts:
alarm_set          20
alarm_query        10
quirky              5
commandstop         2
calendar_remove     1
weather_query       1
lists_remove        1
hue_lightoff        1



intent: alarm_set, total count: 292, total incorrect count: 41
 example of correctly predicted utterance: wake me up at [time : five am] [date : this week]
 example of incorrectly predicted utterance: alert me at [time : three pm] to goto the [event_name : concert]
 
 NOTES: The incorrect example says 'alert me' and mentions an event. Perhaps it is best to move intents events into the calendar.
 Also we can say the words: alert, reminder, remind me, etc. are for calendar. 

incorrect predicted intents for alarm_set and their counts:
calendar_set      29
datetime_query     4
alarm_remove       3
alarm_query        2
calendar_query     2
sendemail          1

intent: alarm_query, total count: 202, total incorrect count: 37
 example of correctly predicted utterance: what alarms i have set
 example of incorrectly predicted utterance: did i set an alarm to [alarm_type : wake up] in the [timeofday : morning]
 NOTES: Do we really want alarm_types or other labels for alarms? This would be confusing. Instead of having utterances like "when is my alarm set to wake up tomorrow" being specific,
 it could be general, giving a list of all of the alarms set for tomorrow. This could be easier. This would cover utterances such as the first example.

incorrect predicted intents for alarm_query and their counts:
alarm_set         22
quirky             5
calendar_query     3
calendar_set       2
datetime_query     2
takeaway_query     1
lists_query        1
email_query        1

