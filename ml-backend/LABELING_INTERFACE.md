# Current config:

```
<View>
  <Header value="Date $recorded_date at $recorded_time in Stable: $stable and Stall: $stall for Horse: $horse"/>
  <Text name="stable" value="Stable: $stable"/>
  <Text name="stall" value="Stall: $stall"/>
  <Text name="horse" value="Horse: $horse"/>
  <Video name="video" value="$video" sync="audio" muted="true" height="280"/>
  <Video name="grid_video" value="$grid_video" sync="audio" muted="true" height="480"/>
  <Audio name="audio" value="$audio" sync="video" hotkey="space" spectrogram="false" decoder="webaudio" defaultscale="10" />
  
  <Header size="4" value="General Labels"/>
   <Labels name="labels" toName="audio" choice="multiple">
		<Label value="Horse Kick" alias="horse_kick" hint="Single impact sound against walls, doors, or objects." background="#8880F8"/>
        <Label value="Rattle Sound / Scraping" alias="rattle" hint="Horses rattle against the cage or metal parts. Or other rattlesounds" background="#fade60ff"/>
		<Label value="Normal" alias="normal" hint="Background shuffling, Normal everyday Barn sounds (brushing, dog bark, birds, etc), quiet standing, low-intensity movement without signs of distress." background="#00bd16"/>
		<Label value="Other Impact Sound" alias="other_impact" hint="Closing Door, Banging, everything that is loud and not produced by the horse." background="#b6bd16"/>
		<Label value="Walking" alias="walking" hint="Hoof clapping sound, so we can distinguish it from the kick." background="#b6bd16"/>
		<Label value="Rolling" alias="rolling" hint="" background="#b16dbd"/>
		<Label value="Unsure" alias="unsure" hint="If the labelling person does not know what something is, but it clearly should be labelled as something." background="#d6bdb6"/>
     	<Label value="Eating" alias="eating" hint="The horse is eating." background="#00DDAA"/>
        <Label value="Pawing" alias="pawing" hint="Pawing is when a horse repeatedly strikes or scrapes the ground (or stall floor) with a front hoof. Also here we label any scraping done by ther horse." background="#974600ff"/>
     	<Label value="Sit down/Stand up" alias="change_stance" hint="When the horse is transitioning from sitting to standing or the other way around." background="#05e0ceff"/>
     	<Label value="Uriante" alias="urinate" hint="Urination detected." background="#ffee00"/>
  </Labels>
  <Header size="4" value="Horse Vocalizations"/>
  <Labels name="horse_labels" toName="audio" choice="multiple">
    <Label value="Neigh / Whinny" alias="neigh" hint="Long, high-pitched call, often for attention or communication." background="#FF8000"/>
    <Label value="Nicker" alias="nicker" hint="Low-pitched, friendly greeting or anticipatory sound (e.g., feeding time)." background="#FF80FF"/>
    <Label value="Squeal" alias="squeal" hint="Short, high-pitched, often in aggressive or defensive interactions." background="#FFF0F0"/>
    <Label value="Snort / Blow" alias="snort" hint="Forceful exhalation; can be alerting, clearing nostrils, or stress-related." background="#D88080"/>
    <Label value="Groan / Moan" alias="groan" hint="Low-frequency vocalization, sometimes linked to discomfort or exertion." background="#DF8FF0"/>
    <Label value="Coughing" alias="coughing" hint="A sudden, forceful expulsion of air from the lungs." background="#1F850F"/>
    <Label value="Heavy breathing / Panting" alias="panting" hint="Rapid, shallow breathing often associated with exertion or stress." background="#AA8D0F"/>
    <Label value="Whimmering (Pain/Distress)" alias="whimmering" hint="Only use if certain! A clear sign that the horse is in distress or pain but unlike a scream it is more quiet." background="#ff0000ff"/>
    <Label value="Scream (Pain/Distress)" alias="scream" hint="Only use if certain! Horse scream is in pain or when scared. See youtube for example." background="#ff0000ff"/>
  </Labels>
  <Header size="4" value="Negative Labels (Trouble-Makers)"/>
  <Labels name="labels_negative" toName="audio" choice="multiple">
    <Label value="Bird Tweet" alias="bird_tweet" hint="Tweeting sounds of birds. Do not add wing flapping here!" background="#ff006aff"/>
    <Label value="Wing flapping" alias="bird_wing_flap" hint="Wing flaps of birds" background="#bbff00ff"/>
  </Labels>
  <TextArea name="uncertainty_reason" toName="audio" perRegion="true" visibleWhenLabel="Unsure" placeholder="Why are you unsure?"/>
  <Header size="4" value="Meta Labels"/>
  <TextArea name="events" toName="audio" value="$events" placeholder="Known events"/> 
  <TextArea name="manual_notes" toName="audio" value="$manual_notes" placeholder="Manual Notes"/>
  <Header size="4" value="Anomaly Detector"/>
  <Labels name="anomaly_labels" toName="audio" choice="multiple">
    <Label value="Anomaly Detector Normal" alias="anomaly_normal" hint="Just everything that is not a horse in distress" background="#d100FF" maxUsages="0"/>
    <Label value="Abnormal (Do not use!)" alias="anomaly_abnormal" hint="DO NOT USE! Only for prelabeling. Please use specific label instead!" background="#d10000" maxUsages="0"/>
  </Labels>
</View>
```


# Example task before import:
[
  {
    "audio": "/data/local-files/?d=label-studio/data/audio/horse123.wav",
    "video": "/data/local-files/?d=label-studio/data/video/horse123_stereo.mp4",
    "stable": "Farm 42",
    "stall": 12
  }
]

# File name convention:
stable##_stall##_horse##_{DATE}_{TIME}_cam.*
stable##_stall##_horse##_{DATE}_{TIME}_mic.*