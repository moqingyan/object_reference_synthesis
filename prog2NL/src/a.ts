type Result = {
    program: [
        ["right", 0, 2],
        ["behind", 1, 0],
        ["material", "rubber", 1]
    ],
    sources: [
        "model",
        "model",
        "exploit"
    ],
    method: {
        type: "exploit",
        depth: 1
    }
};

type material =
    | Rubber [@name "rubber"]
    | Metal [@name "metal"]
[@@deriving yojson]

type clause =
| Right of int * int           [@name "right"]
| Material of matrerial * int  [@name "imperial"]
[@@deriving yojson]

[
    [
        ["right",0,2],
        ["behind",1,0],
        [
            ["material","rubber",1]
        ]
    ]
]