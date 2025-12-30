# Rootstar BCI Hardware Supplies List

Complete shopping list for building the Rootstar BCI hardware platform.

## Quick Reference

**Estimated Total Cost:** $250-600 (depending on electrode quality and optional components)

| Category | Essential | Optional |
|----------|-----------|----------|
| Core Electronics | $55-85 | - |
| fNIRS Optics | $25-50 | - |
| EEG Electrodes | $75-195 | Electrode cap ($50-200) |
| 3D Printed Headset | - | ~$100 (DIY alternative to cap) |
| Neurostimulation | - | $45-85 |
| Passive Components | $40-50 | - |

---

## Core Electronics

### Microcontroller

| Component | Part Number | Description | Qty | Price | Links |
|-----------|-------------|-------------|-----|-------|-------|
| ESP32 Dev Board | **ESP32-WROOM-DA** | Dual-core 240MHz, WiFi/BT, 4MB flash | 1 | $10-15 | [Espressif](https://www.espressif.com/), [DigiKey](https://www.digikey.com/), [Amazon](https://amazon.com/) |

**Notes:**
- The ESP32-WROOM-DA variant has dual antennas for improved connectivity
- Any ESP32-WROOM-32 based dev board will work
- Look for boards with USB-C and built-in USB-to-UART

### EEG Analog-to-Digital Converter

| Component | Part Number | Description | Qty | Price | Links |
|-----------|-------------|-------------|-----|-------|-------|
| EEG ADC | **TI ADS1299** | 8-channel, 24-bit, low-noise biosignal ADC | 1 | $40-60 | [TI](https://www.ti.com/product/ADS1299), [Mouser](https://www.mouser.com/), [DigiKey](https://www.digikey.com/) |

**Specifications:**
- Resolution: 24-bit
- Channels: 8 differential inputs
- Sample rates: 250, 500, 1000, 2000 SPS
- Programmable gain: 1x, 2x, 4x, 6x, 8x, 12x, 24x
- Input-referred noise: 1 uVpp @ 24x gain
- Interface: SPI (up to 20 MHz)
- Built-in bias drive (DRL) and lead-off detection

**Recommended Breakout Boards:**
- OpenBCI Cyton (uses ADS1299) - more expensive but complete
- Custom PCB with ADS1299IPAG (64-pin TQFP)

### fNIRS Analog-to-Digital Converter

| Component | Part Number | Description | Qty | Price | Links |
|-----------|-------------|-------------|-----|-------|-------|
| fNIRS ADC | **TI ADS1115** | 16-bit, 4-channel, I2C ADC | 1 | $5-10 | [TI](https://www.ti.com/product/ADS1115), [Adafruit](https://www.adafruit.com/product/1085), [Amazon](https://amazon.com/) |

**Specifications:**
- Resolution: 16-bit
- Channels: 4 single-ended or 2 differential
- Sample rate: Up to 860 SPS
- Programmable gain: 2/3x to 16x
- Range: +/- 6.144V to +/- 0.256V
- Interface: I2C (up to 3.4 MHz)

---

## fNIRS Optical Components

### Near-Infrared LEDs

| Component | Wavelength | Description | Qty | Price | Links |
|-----------|------------|-------------|-----|-------|-------|
| NIR LED | **760nm** | Deoxyhemoglobin (HbR) sensitive | 2 | $5-10 | [Thorlabs](https://www.thorlabs.com/), [DigiKey](https://www.digikey.com/), [Mouser](https://www.mouser.com/) |
| NIR LED | **850nm** | Oxyhemoglobin (HbO2) sensitive | 2 | $5-10 | [Thorlabs](https://www.thorlabs.com/), [DigiKey](https://www.digikey.com/), [Mouser](https://www.mouser.com/) |

**Recommended Parts:**
- Vishay VSMY2850G (850nm, 3mm)
- Vishay TSAL6100 (940nm alternative)
- OSI Optoelectronics LED760/850

**Notes:**
- Look for LEDs with narrow spectral bandwidth (<50nm FWHM)
- Higher radiant intensity (mW/sr) improves signal quality
- Consider LEDs with built-in lens for better light coupling

### Photodetectors

| Component | Part Number | Description | Qty | Price | Links |
|-----------|-------------|-------------|-----|-------|-------|
| Photodetector | **OPT101** | Monolithic photodiode + transimpedance amp | 2 | $8-15 | [TI](https://www.ti.com/product/OPT101), [DigiKey](https://www.digikey.com/), [Mouser](https://www.mouser.com/) |

**Specifications:**
- Integrated photodiode and transimpedance amplifier
- Responsivity: 0.45 A/W @ 650nm
- Output: Voltage proportional to light intensity
- Bandwidth: 14 kHz (default), adjustable with external capacitor
- Operating voltage: 2.7V to 36V

**Alternative Options:**
- Hamamatsu S1223 series (higher sensitivity)
- Vishay TEMD5080X01 + external TIA

---

## EEG Electrodes

### Wet Electrodes (Recommended for Best Signal Quality)

| Component | Specification | Description | Qty | Price | Links |
|-----------|---------------|-------------|-----|-------|-------|
| EEG Electrodes | **Ag/AgCl, 10mm cup** | Reusable wet electrodes | 8+ | $50-100 | [OpenBCI](https://shop.openbci.com/), [Grass Technologies](https://natus.com/) |
| Reference Electrode | **Ag/AgCl ear clip** | For REF connection | 1 | $5-10 | [OpenBCI](https://shop.openbci.com/) |
| Ground Electrode | **Ag/AgCl ear clip or mastoid** | For GND/DRL connection | 1 | $5-10 | [OpenBCI](https://shop.openbci.com/) |
| Electrode Gel | **Ten20, Signagel** | Conductive paste | 1 | $15-25 | [Amazon](https://amazon.com/), [Weaver and Company](https://weaverandcompany.com/) |
| Electrode Cap | **10-20 system layout** | Pre-positioned electrode holder | 1 | $50-200 | [OpenBCI](https://shop.openbci.com/), [EMOTIV](https://www.emotiv.com/) |

### Dry Electrodes (More Convenient, Slightly Lower Quality)

| Component | Specification | Description | Qty | Price | Links |
|-----------|---------------|-------------|-----|-------|-------|
| Dry EEG Electrodes | **Gold-plated or Ag/AgCl** | No gel required | 8+ | $80-150 | [Florida Research Instruments](https://fri-fl-shop.com/), [OpenBCI](https://shop.openbci.com/) |

### Electrode Placement (10-20 System)

```
        Nasion
           |
    Fp1----Fpz----Fp2
     |      |      |
    F7--F3--Fz--F4--F8
     |      |      |
    T3--C3--Cz--C4--T4
     |      |      |
    T5--P3--Pz--P4--T6
     |      |      |
    O1-----Oz-----O2
           |
        Inion
```

**This system uses 8 channels:**
| Channel | Position | Location |
|---------|----------|----------|
| 0 | Fp1 | Left prefrontal |
| 1 | Fp2 | Right prefrontal |
| 2 | C3 | Left central (motor) |
| 3 | C4 | Right central (motor) |
| 4 | P3 | Left parietal |
| 5 | P4 | Right parietal |
| 6 | O1 | Left occipital |
| 7 | O2 | Right occipital |

---

## 3D Printed Headset (DIY Option)

Instead of purchasing a pre-made electrode cap, you can 3D print a custom headset. This section covers two popular form factors.

### Form Factor Options

#### Option A: Adjustable Band Style

```
        ┌─────────────────────────┐
        │    Electrode Arms       │
        │   ╱   ╱   ╱   ╱   ╱     │
   ┌────┴───────────────────┴────┐
   │  ══════════════════════════ │  ← Adjustable headband
   │  ══════════════════════════ │
   └────┬───────────────────┬────┘
        │   ╲   ╲   ╲   ╲   │
        │    Spring-loaded  │
        │    electrode arms │
        └───────────────────┘
```

**Characteristics:**
- Flexible sizing fits most head sizes
- Electrode arms swing out to 10-20 positions
- Spring-loaded contacts maintain pressure
- Electronics module clips to back
- Easier to print and assemble
- Better for development/prototyping

#### Option B: Ergonomic Shell Style

```
         ┌─────────────┐
        ╱               ╲
       │   ┌─────────┐   │
       │   │ Forehead│   │  ← Rigid contoured shell
       │   │ sensors │   │
        ╲  └─────────┘  ╱
         ╲    ╭───╮    ╱
          ╲   │PCB│   ╱     ← Integrated electronics
           ╲  ╰───╯  ╱
            ╲ ┌───┐ ╱
             ╲│Ear│╱        ← Ear hooks for stability
              └───┘
```

**Characteristics:**
- Clean, consumer-product appearance
- Contoured to head shape
- Fixed electrode positions
- Integrated electronics housing
- Requires head size variants (S/M/L)
- Better for finished product

### 3D Printing Materials

| Component | Recommended Material | Properties |
|-----------|---------------------|------------|
| Main frame | **PETG** | Flexible, durable, skin-safe |
| Electrode holders | **TPU (95A)** | Soft, conforms to scalp |
| Electronics enclosure | **PLA** or **PETG** | Rigid, easy to print |
| Adjustment mechanisms | **PETG** | Strong, slight flexibility |

**Print Settings (PETG):**
- Layer height: 0.2mm
- Infill: 20-30%
- Walls: 3 perimeters
- Nozzle temp: 230-250C
- Bed temp: 70-80C

### Additional Hardware for 3D Printed Headset

| Component | Specification | Qty | Price |
|-----------|---------------|-----|-------|
| PETG Filament | 1kg spool | 1 | $25 |
| TPU Filament | 250g (for electrode pads) | 1 | $15 |
| Compression Springs | 5mm OD x 10mm, light force | 10 | $8 |
| M3 Screws | Assorted lengths (6-20mm) | 50 | $6 |
| M3 Heat-Set Inserts | Brass, 4mm length | 20 | $8 |
| M3 Nuts | Standard | 20 | $3 |
| Elastic Strap | 25mm width, adjustable | 1m | $5 |
| Buckle/Clip | Side-release, 25mm | 2 | $3 |
| Foam Padding | Self-adhesive, 3mm thick | 1 sheet | $5 |
| Shielded Cable | 8-conductor, thin gauge | 2m | $12 |
| Snap Connectors | 4mm snap for electrodes | 10 | $8 |

**Additional cost for DIY headset: ~$100**

### Electrode Mounting Options

#### Spring-Loaded Pin Contacts (Dry)

```
    ┌───────────┐
    │  Holder   │
    │  (PETG)   │
    ├───────────┤
    │  ╔═════╗  │
    │  ║Spring║  │  ← Compression spring
    │  ╚══╤══╝  │
    │     │     │
    └─────┼─────┘
          │
    ┌─────┴─────┐
    │  Contact  │    ← Gold-plated or Ag/AgCl tip
    │   (pin)   │
    └───────────┘
```

| Component | Part | Qty | Price |
|-----------|------|-----|-------|
| Pogo pins | P75-B1, gold-plated, 1mm tip | 10 | $10 |
| Alternative | Spring-loaded test probes | 10 | $15 |

#### Snap-In Cup Holders (Wet)

```
    ┌───────────────┐
    │   Holder      │
    │   (PETG)      │
    │  ┌─────────┐  │
    │  │ ○ snap  │  │  ← 4mm snap connector
    │  │ recess  │  │
    │  └─────────┘  │
    └───────────────┘
          ↓
    ┌─────────────┐
    │  Standard   │
    │  Ag/AgCl    │    ← Use with conductive gel
    │  cup elec.  │
    └─────────────┘
```

#### Comb Electrodes (Dry, Through Hair)

```
       │ │ │ │ │
       │ │ │ │ │      ← Conductive tines
    ┌──┴─┴─┴─┴─┴──┐     (penetrate hair)
    │             │
    │   Contact   │   ← Connection point
    │    base     │
    └─────────────┘
```

Print comb electrodes in conductive PLA or coat with conductive paint/epoxy.

| Component | Specification | Qty | Price |
|-----------|---------------|-----|-------|
| Conductive PLA | Proto-pasta or similar | 100g | $20 |
| OR Conductive paint | Nickel or silver-based | 1 bottle | $15 |

### Design Dimensions (10-20 System)

**Head Measurement Reference:**

```
Circumference measurements:
┌─────────────────────────────────────┐
│  Size    │ Circumference │ Nasion-Inion │
├──────────┼───────────────┼──────────────┤
│  Small   │   52-54 cm    │   32-34 cm   │
│  Medium  │   55-57 cm    │   35-36 cm   │
│  Large   │   58-60 cm    │   37-38 cm   │
└─────────────────────────────────────┘
```

**Electrode Position Calculations:**

The 10-20 system uses percentage-based positioning:

```
Nasion ──────────────────────────────── Inion
   │                                      │
   │  10%   20%   20%   20%   20%   10%   │
   │   │     │     │     │     │     │    │
   ├───┼─────┼─────┼─────┼─────┼─────┼────┤
   Fp  F     C     P     O    (reference points)
```

For an 8-channel system:
- **Fp1/Fp2**: 10% from nasion, 5% lateral from midline
- **C3/C4**: 50% from nasion (vertex), 20% lateral
- **P3/P4**: 70% from nasion, 20% lateral
- **O1/O2**: 90% from nasion, 5% lateral

### Assembly Tips for 3D Printed Headset

1. **Print test fits first** - Print small sections to verify dimensions
2. **Use heat-set inserts** - Stronger than threading directly into plastic
3. **Add foam padding** - Comfort and light blocking at forehead
4. **Route wires internally** - Design channels in the frame for clean cable management
5. **Use strain relief** - Where cables exit the headset
6. **Make electrodes removable** - Easier to clean and replace
7. **Consider modularity** - Separate front/back sections for easier printing

### Electronics Housing Options

**Option 1: Back-of-head module**
```
┌──────────────────┐
│   ┌──────────┐   │
│   │  ESP32   │   │
│   │  ADS1299 │   │  ← Single enclosure
│   │  Battery │   │
│   └──────────┘   │
└──────────────────┘
```

**Option 2: Distributed (belt + headset)**
```
Headset:          Belt clip:
┌────────┐        ┌──────────┐
│ADS1299 │───────→│  ESP32   │
│only    │ cable  │  Battery │
└────────┘        └──────────┘
```

Distributed option reduces headset weight and EMI from digital electronics.

### fNIRS Optode Integration

The fNIRS optical components (LEDs and photodetectors) can be integrated into both headset form factors.

#### fNIRS Design Requirements

| Requirement | Specification | Notes |
|-------------|---------------|-------|
| Source-detector separation | **25-40mm** | Determines penetration depth (~half the separation) |
| Light isolation | Opaque housing | Prevents direct LED→detector path |
| Ambient light blocking | Foam seal + opaque material | Critical for signal quality |
| Scalp contact | Flush or spring-loaded | Air gaps drastically reduce signal |

#### Optode Placement

For combined EEG + fNIRS, the forehead (prefrontal cortex) is ideal for fNIRS:

```
        Nasion
           │
      ┌────┴────┐
     Fp1 [fNIRS] Fp2    ← fNIRS between Fp1/Fp2
      │  ●──○   │         ● = LED, ○ = Photodetector
      │  30mm   │         Source-detector: 30mm
      │         │
     C3   Cz   C4       ← EEG electrodes
      │         │
     P3   Pz   P4
      │         │
     O1   Oz   O2
           │
        Inion
```

**Why forehead?**
- Less hair interference
- Thinner skull (~6mm vs 10mm elsewhere)
- Prefrontal cortex access (attention, executive function)
- Natural integration point for both form factors

#### Form Factor A: Modular Optode Pod (Adjustable Band)

```
      fNIRS Forehead Pod
    ┌─────────────────────┐
    │  ┌─────┐   ┌─────┐  │
    │  │ LED │   │ PD  │  │  ← 30mm separation
    │  │ 760 │   │OPT  │  │
    │  │ 850 │   │101  │  │
    │  └──┬──┘   └──┬──┘  │
    │     │ wires  │      │
    └─────┼────────┼──────┘
          │        │
    ══════╧════════╧══════  ← Clips to headband
         Adjustable Band
```

**Advantages:**
- Independent positioning from EEG electrodes
- Easy to remove for cleaning
- Can adjust source-detector separation
- Simpler to prototype

#### Form Factor B: Integrated Optodes (Ergonomic Shell)

```
         ┌─────────────────┐
        ╱   ●    30mm    ○  ╲   ← Built into forehead
       │   LED          PD   │     section of shell
       │   ├────────────┤    │
       │   │  Light     │    │
       │   │  barrier   │    │   ← Internal wall blocks
        ╲  │  (opaque)  │   ╱       direct light path
         ╲ └────────────┘  ╱
          ╲               ╱
           └─────────────┘
```

**Advantages:**
- Cleaner appearance
- Fixed, repeatable positioning
- Better light sealing
- More professional look

#### Optode Housing Cross-Section

```
        Opaque shell (black PETG)
    ┌─────────────────────────────┐
    │  ████████████████████████   │
    │  ██┌───────────────┐████   │
    │  ██│               │████   │  ← Component cavity
    │  ██│   LED or PD   │████   │
    │  ██│               │████   │
    │  ██└───────┬───────┘████   │
    │  ██████████│████████████   │
    └────────────┼────────────────┘
                 │
    ┌────────────┴────────────┐
    │   Translucent TPU pad   │    ← Light passes through
    │   (conforms to skin)    │       Soft contact
    └─────────────────────────┘
                 │
    ─────────────┴─────────────  Scalp
```

#### Additional Hardware for fNIRS Integration

| Component | Specification | Qty | Price |
|-----------|---------------|-----|-------|
| Black PETG | Light-blocking housing | 100g | $8 |
| Translucent TPU | Optical window/skin contact | 50g | $8 |
| Light barrier foam | Closed-cell, self-adhesive | 1 sheet | $5 |
| Heat shrink (black) | Wire light-blocking | 1m | $3 |

**Additional cost for fNIRS integration: ~$25**

#### fNIRS Wiring Considerations

```
LED Circuit:
    ESP32 GPIO 25/26/27
           │
           ▼
    ┌──────────────┐
    │   Current    │    ← Limit to ~20mA per LED
    │   Limiting   │       Use PWM for intensity control
    │   Resistor   │
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │  NIR LED     │
    │  760/850nm   │
    └──────────────┘

Photodetector Circuit:
    ┌──────────────┐
    │   OPT101     │    ← Built-in transimpedance amp
    │              │       Output: 0-5V proportional to light
    └──────┬───────┘
           │
           ▼
    ADS1115 ADC (I2C)
           │
           ▼
       ESP32
```

#### Light Isolation Tips

1. **Use black filament** for all optode housings
2. **Add internal baffles** between LED and photodetector cavities
3. **Seal edges with foam** to block ambient light
4. **Use black heat shrink** on wires near optodes
5. **Test in bright light** - signal should be stable

#### Recommended fNIRS Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| LED drive current | 20mA | Balance brightness vs heating |
| PWM frequency | 1kHz+ | Above flicker perception |
| Sampling rate | 10-50 Hz | Hemodynamics are slow |
| Source-detector distance | 30mm | Good cortical penetration |
| Measurement cycle | 760nm → read → 850nm → read | Time-division multiplexing |

---

## Neurostimulation Components (Optional)

**WARNING: Neurostimulation carries inherent risks. Only use with proper safety circuits and within established safety limits. Consult medical professionals before any human testing.**

### Digital-to-Analog Converter

| Component | Part Number | Description | Qty | Price | Links |
|-----------|-------------|-------------|-----|-------|-------|
| DAC | **DAC8564** | 4-channel, 16-bit, SPI DAC | 1 | $15-25 | [TI](https://www.ti.com/product/DAC8564), [DigiKey](https://www.digikey.com/) |

### Current Source Circuit

| Component | Description | Qty | Price |
|-----------|-------------|-----|-------|
| Op-Amp | OPA2277 or similar precision op-amp | 2 | $5-10 |
| Precision Resistors | 1% tolerance, various values | 10 | $5 |
| Current Sense Resistor | 0.1% tolerance | 1 | $3 |

**Howland Current Source Notes:**
- Converts voltage output to controlled current
- Hardware current limiting is MANDATORY
- Maximum 2mA output (safety limit)
- Add hardware comparator for overcurrent shutdown

### Stimulation Electrodes

| Component | Specification | Description | Qty | Price | Links |
|-----------|---------------|-------------|-----|-------|-------|
| Stim Electrodes | **Saline sponge, 25-35 cm2** | tDCS/tACS electrodes | 2 | $20-40 | [Soterix Medical](https://soterixmedical.com/), [Neuroelectrics](https://www.neuroelectrics.com/) |

---

## Passive Components & Wiring

### Power Supply Filtering

| Component | Value | Package | Qty | Price |
|-----------|-------|---------|-----|-------|
| Ceramic Capacitor | 100nF | 0805 SMD or through-hole | 10 | $2 |
| Electrolytic Capacitor | 10uF | Through-hole | 5 | $2 |
| Electrolytic Capacitor | 100uF | Through-hole | 2 | $2 |

### Resistors

| Component | Values | Package | Qty | Price |
|-----------|--------|---------|-----|-------|
| Resistor Kit | 10R - 1M, 1% | Through-hole | 1 kit | $5 |

### Connectors & Wiring

| Component | Specification | Description | Qty | Price |
|-----------|---------------|-------------|-----|-------|
| USB Cable | USB-A to Micro-B or USB-C | ESP32 programming/power | 1 | $5 |
| Jumper Wires | Dupont, M-M, M-F, F-F | Prototyping | 40+ | $5 |
| Pin Headers | 2.54mm pitch | For breakout boards | 2 | $2 |
| Breadboard | 830 tie-points | Full-size | 1 | $5-10 |

### Logic Level Shifting (If Needed)

| Component | Part Number | Description | Qty | Price |
|-----------|-------------|-------------|-----|-------|
| Level Shifter | TXB0108 or BSS138 | 3.3V to 5V bidirectional | 1 | $3-5 |

---

## GPIO Wiring Reference

### ESP32 Pin Assignments

```
┌─────────────────────────────────────────────────────────────┐
│                      ESP32-WROOM-DA                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  SPI Bus (ADS1299 EEG):                                      │
│    GPIO 23 ──► MOSI (Data to ADC)                           │
│    GPIO 19 ◄── MISO (Data from ADC)                         │
│    GPIO 18 ──► SCLK (SPI Clock)                             │
│    GPIO 5  ──► CS   (Chip Select, active low)               │
│    GPIO 4  ◄── DRDY (Data Ready interrupt)                  │
│                                                              │
│  I2C Bus (ADS1115 fNIRS):                                    │
│    GPIO 21 ◄─► SDA  (I2C Data)                              │
│    GPIO 22 ──► SCL  (I2C Clock)                             │
│                                                              │
│  PWM Outputs (NIR LEDs):                                     │
│    GPIO 25 ──► LED 760nm #1                                 │
│    GPIO 26 ──► LED 760nm #2                                 │
│    GPIO 27 ──► LED 850nm                                    │
│                                                              │
│  DAC Outputs (Stimulation):                                  │
│    GPIO 32 ──► DAC Channel A                                │
│    GPIO 33 ──► DAC Channel B                                │
│                                                              │
│  Status:                                                     │
│    GPIO 17 ──► Status LED                                   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Connection Diagram

```
                    ┌──────────────┐
                    │   ESP32      │
                    │   WROOM-DA   │
                    └──────┬───────┘
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
         ▼                 ▼                 ▼
    ┌─────────┐      ┌──────────┐      ┌─────────┐
    │ ADS1299 │      │ ADS1115  │      │ DAC8564 │
    │  (SPI)  │      │  (I2C)   │      │  (SPI)  │
    └────┬────┘      └────┬─────┘      └────┬────┘
         │                │                 │
         ▼                ▼                 ▼
    ┌─────────┐      ┌──────────┐      ┌─────────┐
    │   EEG   │      │  fNIRS   │      │  Stim   │
    │Electrodes│     │ Optics   │      │ Output  │
    └─────────┘      └──────────┘      └─────────┘
```

---

## Safety Limits (Built into Firmware)

| Parameter | Limit | Enforcement |
|-----------|-------|-------------|
| Maximum stimulation current | 2 mA (2000 uA) | Hardware + Software |
| Maximum stimulation duration | 30 minutes | Software timer |
| Maximum tACS frequency | 100 Hz | Software limit |
| Minimum ramp time | 10 ms | Software enforced |
| Current sensing | Continuous | Hardware ADC feedback |

**CRITICAL:** Software limits are NOT sufficient for safety. Always implement:
1. Hardware current limiting (resistor + comparator)
2. Fuse protection
3. Hardware watchdog timer
4. Physical emergency stop

---

## Recommended Suppliers

| Supplier | Specialty | Website |
|----------|-----------|---------|
| DigiKey | General electronics | https://www.digikey.com/ |
| Mouser | General electronics | https://www.mouser.com/ |
| Adafruit | Breakout boards, dev kits | https://www.adafruit.com/ |
| SparkFun | Breakout boards, tutorials | https://www.sparkfun.com/ |
| OpenBCI | BCI-specific hardware | https://shop.openbci.com/ |
| Thorlabs | Optical components | https://www.thorlabs.com/ |
| Texas Instruments | ICs, eval boards | https://www.ti.com/ |

---

## Assembly Notes

1. **Start with the ESP32** - Verify basic functionality before adding other components
2. **Add ADS1115 first** - I2C is simpler to debug than SPI
3. **Test fNIRS optics** - Verify LED/photodetector response before full integration
4. **Add ADS1299 last** - Most complex component, requires proper analog layout
5. **Use star grounding** - Single ground point for all analog circuits
6. **Shield analog signals** - Use shielded cables for electrode connections
7. **Separate power supplies** - Digital and analog should have separate regulation

## Troubleshooting

| Issue | Possible Cause | Solution |
|-------|----------------|----------|
| No SPI communication | Wrong GPIO or clock speed | Verify wiring, reduce SPI clock |
| Noisy EEG signal | Poor grounding, interference | Check ground, add shielding |
| fNIRS signal drift | Ambient light, LED heating | Use optical isolation, add bandpass filter |
| ESP32 brownout | Insufficient power | Use powered USB hub, add decoupling caps |
