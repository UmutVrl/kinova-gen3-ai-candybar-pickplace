##########################################################
#    Kinova Gen3 Robotic Arm                             #
#                                                        #
#    Skeleton                                            #
#                                                        #
#                                                        #
#    written by: U. Vural                                #
#                                                        #
#                                                        #
#    for KISS Project at Furtwangen University           #
#    (06.2025)                                           #
##########################################################

#[Start stream & display processes]
#        |
#[Move to retract position & open gripper]
#        |
#[Move to start position]
#        |
#[Start detection process]
#        |
#[Wait up to 10 seconds for detection]
#        |
#   /----------------------------\
#[Detected?]                  [Not detected?]
#   |                              |
#[Save coords, move to obj]   [Return to retract]
 #       |
#[Cleanup, shutdown, user quit]