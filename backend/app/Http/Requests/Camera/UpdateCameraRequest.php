<?php

namespace App\Http\Requests\Camera;

use Illuminate\Foundation\Http\FormRequest;

class UpdateCameraRequest extends FormRequest
{
    public function authorize(): bool { return true; }

    public function rules(): array
    {
        return [
            'location'   => 'sometimes|string|max:255',
            'isactive'   => 'sometimes|boolean',
            'code'       => 'sometimes|nullable|string|max:64|unique:camera,code,' . $this->route('id') . ',camera_id',
            'stream_url' => 'sometimes|nullable|string|max:512',
        ];
    }
}
