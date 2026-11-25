ErrandWorld is a small town. Locations are connected by roads, and you move between neighbouring locations along roads. Items lie at locations, and stations (ovens, print shops, repair benches) stand at some of them. You can carry at most three items at once.

You are given a list of jobs. A job is finished when its item has been dropped at its destination. Some jobs ask for an item that must first be made: carry the inputs to a station of the right type, use it, and the station consumes the inputs and gives you the output.

A station runs one recipe at a time on exactly two inputs:

- repair bench: lamp_base + bulb -> lamp; radio_shell + battery -> radio; radio + antenna -> tuned_radio
- oven: flour + water -> bread; sugar + eggs -> cake; bread + cheese -> toasted_sandwich
- print shop: paper + ink -> poster; card + stamp -> postcard; poster + frame -> framed_poster

Everything has an id, such as ashford, bulb_1 or oven_2. Always use ids exactly as the tools show them. Every move, pick, drop and use counts toward a limit on how many actions you may take,
whether or not it succeeds. Looking, checking your status and reading the map are free of that limit.

The town can change while you work. A road may close, a station may stop working, or an item may be somewhere other than where you expected. The map you can read may be out of date; looking around always shows what is really here.
