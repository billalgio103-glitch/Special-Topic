PRAGMA foreign_keys = ON;
CREATE TABLE sources (
    source_id TEXT PRIMARY KEY,
    source_url TEXT NOT NULL,
    revision TEXT NOT NULL,
    license TEXT NOT NULL,
    expected_samples INTEGER NOT NULL CHECK(expected_samples > 0)
);
CREATE TABLE assets (
    asset_path TEXT PRIMARY KEY,
    source_url TEXT NOT NULL,
    bytes INTEGER NOT NULL CHECK(bytes > 0),
    sha256 TEXT NOT NULL CHECK(length(sha256) = 64)
);
CREATE TABLE images (
    image_id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL REFERENCES sources(source_id),
    source_key TEXT NOT NULL,
    image_asset TEXT NOT NULL REFERENCES assets(asset_path),
    mask_asset TEXT NOT NULL REFERENCES assets(asset_path),
    array_index INTEGER,
    binary_mask_path TEXT NOT NULL UNIQUE,
    binary_mask_sha256 TEXT NOT NULL,
    width INTEGER NOT NULL CHECK(width > 0),
    height INTEGER NOT NULL CHECK(height > 0),
    original_dtype TEXT NOT NULL,
    mask_type TEXT NOT NULL CHECK(mask_type IN ('instance','binary','one_hot')),
    object_kind TEXT NOT NULL CHECK(object_kind IN ('particle_instance','connected_component')),
    pixel_size_x_nm REAL CHECK(pixel_size_x_nm > 0),
    pixel_size_y_nm REAL CHECK(pixel_size_y_nm > 0),
    calibration_status TEXT NOT NULL,
    group_id TEXT NOT NULL,
    split TEXT NOT NULL CHECK(split IN ('train','validation','test')),
    split_note TEXT NOT NULL,
    material TEXT,
    nominal_particle_size_nm REAL,
    intensity_mean REAL NOT NULL,
    intensity_std REAL NOT NULL CHECK(intensity_std >= 0),
    foreground_pixels INTEGER NOT NULL CHECK(foreground_pixels >= 0),
    object_count INTEGER NOT NULL CHECK(object_count >= 0),
    source_metadata_json TEXT NOT NULL,
    UNIQUE(source_id, source_key),
    CHECK(foreground_pixels <= width * height),
    CHECK((pixel_size_x_nm IS NULL) = (pixel_size_y_nm IS NULL))
);
CREATE TABLE objects (
    image_id TEXT NOT NULL REFERENCES images(image_id),
    label_id INTEGER NOT NULL CHECK(label_id > 0),
    area_px INTEGER NOT NULL CHECK(area_px > 0),
    area_nm2 REAL CHECK(area_nm2 > 0),
    equivalent_diameter_px REAL NOT NULL CHECK(equivalent_diameter_px > 0),
    equivalent_diameter_nm REAL CHECK(equivalent_diameter_nm > 0),
    touches_border INTEGER NOT NULL CHECK(touches_border IN (0,1)),
    PRIMARY KEY(image_id, label_id)
);
CREATE INDEX images_source_split ON images(source_id, split);
CREATE INDEX images_group ON images(group_id);
CREATE VIEW dataset_summary AS
SELECT source_id, split, COUNT(*) AS image_count, SUM(object_count) AS object_count,
       AVG(1.0 * foreground_pixels / (width * height)) AS mean_foreground_fraction,
       SUM(pixel_size_x_nm IS NOT NULL) AS calibrated_images
FROM images GROUP BY source_id, split;
-- Instance masks identify particles; components in binary masks can merge touching particles.
CREATE VIEW size_measurements AS
SELECT i.source_id, i.split, i.object_kind, i.material, i.calibration_status, o.*
FROM objects o JOIN images i USING(image_id) WHERE o.touches_border = 0;
CREATE VIEW particle_sizes_nm AS
SELECT * FROM size_measurements
WHERE object_kind = 'particle_instance' AND equivalent_diameter_nm IS NOT NULL;
